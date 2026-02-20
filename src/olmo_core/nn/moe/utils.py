import os
from typing import Optional, Tuple, cast

import torch

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _HAS_TRITON = True
except Exception:  # pragma: no cover - import guard
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

from olmo_core.kernels.moe_chunk_reorder import moe_chunk_permute, moe_chunk_unpermute
from olmo_core.kernels.moe_permute_drop import moe_permute_drop_fwd
from olmo_core.utils import get_or_init_stream

import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.permutation import moe_permute, moe_sort_chunks_by_index, moe_unpermute


# LAST_STREAM_ID = None
@torch.compiler.disable  # helper runs eagerly,
def async_copy_to_cpu(
    gpu_buf, event=None, return_event=True
) -> Tuple[torch.Tensor, torch.cuda.Stream, Optional[torch.cuda.Event]]:
    # *** async copy to CPU for future GroupedGEMM ***
    # start a new stream for the copy
    dtoh_stream = get_or_init_stream(id="dtoh", priority=-5)  # TODO: check any id that's not 0?

    # global LAST_STREAM_ID
    # if LAST_STREAM_ID is None:
    #     LAST_STREAM_ID = id(dtoh_stream)
    #     print(f"Initialized LAST_STREAM_ID: {LAST_STREAM_ID}")
    # else:
    #     assert LAST_STREAM_ID == id(dtoh_stream), f"Expected stream id {LAST_STREAM_ID}, got {id(dtoh_stream)}"

    # Make the copy_stream start after everything already queued on the
    # current stream (default) that touches batch_size_per_expert.
    dtoh_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(dtoh_stream):
        cpu_buf = torch.empty_like(
            gpu_buf, device="cpu", pin_memory=True
        )  # compile does not work with pin_memory
        cpu_buf.copy_(gpu_buf, non_blocking=True)

    dtoh_event = dtoh_stream.record_event(event)
    dtoh_event = cast(torch.cuda.Event, dtoh_event)

    # cpu_buf = gpu_buf.to(torch.device("cpu"), non_blocking=True)
    # Keep the source tensor alive until the copy_stream is done
    # gpu_buf.record_stream(dtoh_stream) # NOTE: does not work with compile
    if return_event:
        return cpu_buf, dtoh_stream, dtoh_event

    return cpu_buf, dtoh_stream, None


@torch.compiler.disable  # helper runs eagerly,
def wait_stream_no_compile(this_stream: torch.cuda.Stream, other_stream: torch.cuda.Stream):
    this_stream.wait_stream(other_stream)


# disable compile for permute
@torch.compiler.disable
def moe_permute_no_compile(*args, **kwargs):
    return moe_permute(*args, **kwargs)


@torch.compiler.disable
def moe_unpermute_no_compile(*args, **kwargs):
    return moe_unpermute(*args, **kwargs)


class _RowPermutationReorderAutograd(torch.autograd.Function):
    """Reorder rows by a permutation; backward uses inverse gather."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        inp: torch.Tensor,
        reorder_indices: torch.Tensor,
        inverse_reorder_indices: torch.Tensor,
        out: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if out is not None:
            output = torch.index_select(inp, 0, reorder_indices, out=out)
            ctx.mark_dirty(output)
        else:
            output = torch.index_select(inp, 0, reorder_indices)

        # Store directly on ctx (matches other autograd wrappers in this module).
        ctx.inverse_reorder_indices = inverse_reorder_indices
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        grad_inp = None
        if ctx.needs_input_grad[0]:
            if not grad_output.is_contiguous():
                grad_output = grad_output.contiguous()
            grad_inp = torch.index_select(grad_output, 0, ctx.inverse_reorder_indices)
        return grad_inp, None, None, None


class _MoePermuteDropIndexMapAutograd(torch.autograd.Function):
    """Custom CUDA one-shot permute+drop with TE-compatible packed row_id_map."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        inp: torch.Tensor,
        routing_map: torch.Tensor,
        requested_splits: torch.Tensor,
        keep_splits: torch.Tensor,
        num_out_tokens: int,
        out: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        requested_i64 = requested_splits.to(dtype=torch.long).contiguous()
        keep_i64 = keep_splits.to(dtype=torch.long).contiguous()
        requested_offsets = torch.cumsum(requested_i64, dim=0) - requested_i64
        keep_offsets = torch.cumsum(keep_i64, dim=0) - keep_i64

        dropped, row_id_map = moe_permute_drop_fwd(
            inp=inp,
            routing_map=routing_map,
            requested_offsets=requested_offsets,
            keep_offsets=keep_offsets,
            keep_splits=keep_i64,
            num_out_tokens=num_out_tokens,
            out=out,
        )
        if out is not None and dropped is out:
            ctx.mark_dirty(dropped)

        ctx.row_id_map = row_id_map
        ctx.num_tokens = routing_map.shape[0]
        ctx.topK = routing_map.shape[1]
        ctx.mark_non_differentiable(row_id_map)
        return dropped, row_id_map

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_row_id_map: torch.Tensor):  # type: ignore[override]
        del grad_row_id_map
        grad_inp = None
        if ctx.needs_input_grad[0]:
            if not grad_output.is_contiguous():
                grad_output = grad_output.contiguous()
            grad_inp = tex.moe_permute_bwd(
                grad_output,
                TE_DType[grad_output.dtype],
                ctx.row_id_map,
                torch.empty(0),
                ctx.num_tokens,
                ctx.topK,
            )
        return grad_inp, None, None, None, None, None



def moe_reorder_rows_permutation(
    *,
    inp: torch.Tensor,
    reorder_indices: torch.Tensor,
    inverse_reorder_indices: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if inp.ndim != 2:
        raise ValueError(f"Expected rank-2 input, got shape={tuple(inp.shape)}")
    if reorder_indices.ndim != 1:
        raise ValueError(f"Expected rank-1 reorder_indices, got shape={tuple(reorder_indices.shape)}")
    if inverse_reorder_indices.ndim != 1:
        raise ValueError(
            "Expected rank-1 inverse_reorder_indices, "
            f"got shape={tuple(inverse_reorder_indices.shape)}"
        )
    if reorder_indices.numel() != inp.shape[0]:
        raise ValueError(
            f"reorder_indices size ({reorder_indices.numel()}) must equal input rows ({inp.shape[0]})"
        )
    if inverse_reorder_indices.numel() != inp.shape[0]:
        raise ValueError(
            f"inverse_reorder_indices size ({inverse_reorder_indices.numel()}) must equal input rows ({inp.shape[0]})"
        )
    if reorder_indices.device != inp.device:
        raise ValueError(
            f"reorder_indices/input device mismatch: reorder_indices={reorder_indices.device} inp={inp.device}"
        )
    if inverse_reorder_indices.device != inp.device:
        raise ValueError(
            "inverse_reorder_indices/input device mismatch: "
            f"inverse_reorder_indices={inverse_reorder_indices.device} inp={inp.device}"
        )
    if out is not None:
        if out.shape != inp.shape:
            raise ValueError(f"out shape mismatch: out={tuple(out.shape)} inp={tuple(inp.shape)}")
        if out.dtype != inp.dtype:
            raise ValueError(f"out dtype mismatch: out={out.dtype} inp={inp.dtype}")
        if out.device != inp.device:
            raise ValueError(f"out device mismatch: out={out.device} inp={inp.device}")

    reorder_i64 = (
        reorder_indices if reorder_indices.dtype == torch.long else reorder_indices.to(dtype=torch.long)
    )
    inverse_i64 = (
        inverse_reorder_indices
        if inverse_reorder_indices.dtype == torch.long
        else inverse_reorder_indices.to(dtype=torch.long)
    )
    reorder_i64 = reorder_i64.contiguous()
    inverse_i64 = inverse_i64.contiguous()
    return _RowPermutationReorderAutograd.apply(inp, reorder_i64, inverse_i64, out)



def moe_permute_1d_fused_drop_no_compile(
    *,
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
    reorder_indices: torch.Tensor,
    inverse_reorder_indices: torch.Tensor,
    requested_splits: Optional[torch.Tensor] = None,
    keep_splits: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    map_type: str = "index",
) -> tuple[torch.Tensor, torch.Tensor]:
    """TE permute + drop-token reorder with optional one-shot custom CUDA backend."""
    if map_type != "index":
        raise ValueError(f"moe_permute_1d_fused_drop_no_compile only supports map_type='index' (got {map_type})")

    if (requested_splits is None) != (keep_splits is None):
        raise ValueError("requested_splits and keep_splits must both be provided for one-shot permute+drop")

    if requested_splits is not None and keep_splits is not None:
        if requested_splits.ndim != 1:
            raise ValueError(f"Expected rank-1 requested_splits, got shape={tuple(requested_splits.shape)}")
        if keep_splits.ndim != 1:
            raise ValueError(f"Expected rank-1 keep_splits, got shape={tuple(keep_splits.shape)}")
        if requested_splits.numel() != keep_splits.numel():
            raise ValueError(
                f"requested_splits/keep_splits size mismatch: {requested_splits.numel()} vs {keep_splits.numel()}"
            )
        if requested_splits.device != inp.device:
            raise ValueError(
                f"requested_splits/input device mismatch: requested_splits={requested_splits.device} inp={inp.device}"
            )
        if keep_splits.device != inp.device:
            raise ValueError(
                f"keep_splits/input device mismatch: keep_splits={keep_splits.device} inp={inp.device}"
            )
        if not inp.is_cuda:
            raise ValueError("one-shot permute+drop path requires CUDA input")
        dropped, row_id_map = _MoePermuteDropIndexMapAutograd.apply(
            inp,
            routing_map,
            requested_splits,
            keep_splits,
            num_out_tokens,
            out,
        )
        return dropped, row_id_map

    reorder_i64 = (
        reorder_indices if reorder_indices.dtype == torch.long else reorder_indices.to(dtype=torch.long)
    )
    inverse_i64 = (
        inverse_reorder_indices
        if inverse_reorder_indices.dtype == torch.long
        else inverse_reorder_indices.to(dtype=torch.long)
    )

    permuted, row_id_map = moe_permute_no_compile(
        inp=inp,
        routing_map=routing_map, # indices of experts. shape: [token_cnt, topK]
        num_out_tokens=num_out_tokens,
        map_type="index",
    )
    dropped = moe_reorder_rows_permutation(
        inp=permuted,
        reorder_indices=reorder_i64, # shape: [token_cnt * topK]
        inverse_reorder_indices=inverse_i64, # shape: [token_cnt * topK]
        out=out,
    )
    return dropped, row_id_map


if _HAS_TRITON:

    @triton.jit
    def _fused_unpermute_row_id_remap_kernel(
        row_id_map_ptr,
        inverse_reorder_ptr,
        num_kept_ptr,
        out_ptr,
        n_rows,
        inverse_n_rows,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        in_bounds = offs < n_rows

        row = tl.load(row_id_map_ptr + offs, mask=in_bounds, other=-1)
        valid_row = (row >= 0) & (row < inverse_n_rows)
        safe_row = tl.where(valid_row, row, 0)

        packed_idx = tl.load(inverse_reorder_ptr + safe_row, mask=in_bounds, other=0)
        num_kept = tl.load(num_kept_ptr)
        valid_kept = packed_idx < num_kept
        out_row = tl.where(valid_row & valid_kept, packed_idx, -1)

        tl.store(out_ptr + offs, out_row.to(tl.int32), mask=in_bounds)


def _normalize_num_kept_tensor(
    *,
    num_kept: Optional[torch.Tensor],
    packed_keep_mask: torch.Tensor,
) -> torch.Tensor:
    if num_kept is None:
        return packed_keep_mask.to(dtype=torch.long).sum(dtype=torch.long).view(1)
    if num_kept.numel() != 1:
        raise ValueError(f"num_kept must contain a single element, got shape={tuple(num_kept.shape)}")
    if num_kept.device != packed_keep_mask.device:
        raise ValueError(
            f"num_kept/passed_keep_mask device mismatch: num_kept={num_kept.device} "
            f"packed_keep_mask={packed_keep_mask.device}"
        )
    return num_kept.to(dtype=torch.long).reshape(1)


def _build_fused_unpermute_row_id_map(
    *,
    row_id_map_i32: torch.Tensor,
    inverse_reorder_i64: torch.Tensor,
    num_kept: torch.Tensor,
) -> torch.Tensor:
    row_id_map_i32 = row_id_map_i32.contiguous()
    inverse_reorder_i64 = inverse_reorder_i64.contiguous()
    num_kept = num_kept.contiguous()

    out = torch.empty_like(row_id_map_i32, dtype=torch.int32)
    n_rows = row_id_map_i32.numel()
    if n_rows == 0:
        return out

    if _HAS_TRITON and row_id_map_i32.is_cuda and inverse_reorder_i64.is_cuda and num_kept.is_cuda:
        block = 1024
        grid = (triton.cdiv(n_rows, block),)
        _fused_unpermute_row_id_remap_kernel[grid](
            row_id_map_i32,
            inverse_reorder_i64,
            num_kept,
            out,
            n_rows,
            inverse_reorder_i64.numel(),
            BLOCK=block,
        )
        return out

    valid = row_id_map_i32 >= 0
    safe_src = torch.where(valid, row_id_map_i32.to(dtype=torch.long), torch.zeros_like(row_id_map_i32, dtype=torch.long))
    packed_idx = inverse_reorder_i64.index_select(0, safe_src)
    fused_valid = valid & (packed_idx < num_kept.view(()))
    return torch.where(
        fused_valid,
        packed_idx.to(dtype=torch.int32),
        torch.full_like(row_id_map_i32, -1),
    )


class _TEUnpermuteIndexMapMaskedAutograd(torch.autograd.Function):
    """TE index-map unpermute with explicit dropped-row grad masking."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        inp: torch.Tensor,
        row_id_map: torch.Tensor,
        merging_probs: torch.Tensor,
        packed_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        if row_id_map.dtype != torch.int32:
            row_id_map = row_id_map.to(dtype=torch.int32)
        probs = merging_probs
        if probs.dtype != torch.float32:
            probs = probs.to(dtype=torch.float32)
        output = tex.moe_unpermute_fwd(
            inp,
            TE_DType[inp.dtype],
            row_id_map,
            probs,
            probs.shape[0],
            probs.shape[1],
        )
        ctx.save_for_backward(inp, row_id_map, probs, packed_keep_mask)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        inp, row_id_map, probs, packed_keep_mask = ctx.saved_tensors
        grad_inp = None
        grad_probs = None
        if ctx.needs_input_grad[0]:
            grad_inp, grad_probs = tex.moe_unpermute_bwd(
                grad_output,
                inp,
                TE_DType[grad_output.dtype],
                row_id_map,
                probs,
            )
            # In TE unpermute backward, act_grad is allocated with torch::empty(...) (not zeroed)
            # TE backward kernel only writes when dest_row != -1 (valid row)
            # So rows corresponding to dropped tokens can remain uninitialized garbage unless you zero them.
            grad_inp.masked_fill_(~packed_keep_mask.unsqueeze(-1), 0)


        if not ctx.needs_input_grad[2]:
            grad_probs = None
        return grad_inp, None, grad_probs, None


# @torch.compiler.disable
def moe_unpermute_1d_fused_drop_no_compile(
    *,
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    local_inverse_reorder_indices: torch.Tensor,
    packed_keep_mask: torch.Tensor,
    merging_probs: torch.Tensor,
    num_kept: Optional[torch.Tensor] = None,
    row_id_map_is_packed: bool = False,
    map_type: str = "index",
) -> torch.Tensor:
    """
    Fused 1D restore-drop + TE index-map unpermute.
    TODO: add a custom CUDA backend mirroring TE `_moe_unpermute_index_map`.
    """
    if map_type != "index":
        raise ValueError(f"moe_unpermute_1d_fused_drop_no_compile only supports map_type='index' (got {map_type})")
    if inp.ndim != 2:
        raise ValueError(f"Expected rank-2 input, got shape={tuple(inp.shape)}")
    if row_id_map.ndim != 1:
        raise ValueError(f"Expected rank-1 row_id_map, got shape={tuple(row_id_map.shape)}")
    if local_inverse_reorder_indices.ndim != 1:
        raise ValueError(
            "Expected rank-1 local_inverse_reorder_indices, "
            f"got shape={tuple(local_inverse_reorder_indices.shape)}"
        )
    if packed_keep_mask.ndim != 1:
        raise ValueError(f"Expected rank-1 packed_keep_mask, got shape={tuple(packed_keep_mask.shape)}")
    if row_id_map.numel() != local_inverse_reorder_indices.numel():
        raise ValueError(
            "row_id_map/local_inverse_reorder_indices size mismatch: "
            f"{row_id_map.numel()} vs {local_inverse_reorder_indices.numel()}"
        )
    if packed_keep_mask.numel() != local_inverse_reorder_indices.numel():
        raise ValueError(
            "packed_keep_mask/local_inverse_reorder_indices size mismatch: "
            f"{packed_keep_mask.numel()} vs {local_inverse_reorder_indices.numel()}"
        )
    if inp.shape[0] != packed_keep_mask.numel():
        raise ValueError(
            f"input rows ({inp.shape[0]}) must equal packed_keep_mask size ({packed_keep_mask.numel()})"
        )
    if num_kept is not None and not torch.is_tensor(num_kept):
        raise ValueError(f"num_kept must be a scalar tensor when provided, got {type(num_kept)}")

    row_map_i32 = row_id_map if row_id_map.dtype == torch.int32 else row_id_map.to(dtype=torch.int32)
    inverse_i64 = (
        local_inverse_reorder_indices
        if local_inverse_reorder_indices.dtype == torch.long
        else local_inverse_reorder_indices.to(dtype=torch.long)
    )
    keep_mask = packed_keep_mask if packed_keep_mask.dtype == torch.bool else packed_keep_mask.to(dtype=torch.bool)
    num_kept_t = _normalize_num_kept_tensor(num_kept=num_kept, packed_keep_mask=keep_mask)

    if row_id_map_is_packed:
        remapped_row_id_map = row_map_i32
    else:
        # Equivalent: packed_idx = inverse_reorder[row_id_map] (for valid rows)
        # keep only entries with packed_idx < num_kept
        # else set to -1
        remapped_row_id_map = _build_fused_unpermute_row_id_map(
            row_id_map_i32=row_map_i32, # TE unpermute map in pre-drop row order)
            inverse_reorder_i64=inverse_i64, # Applies inverse_reorder_i64 to convert each referenced row into packed order (kept rows first, dropped rows after).
            num_kept=num_kept_t,
        )

    return _TEUnpermuteIndexMapMaskedAutograd.apply(inp, remapped_row_id_map, merging_probs, keep_mask)


@torch.compiler.disable
def moe_sort_chunks_by_index_no_compile(*args, **kwargs):
    return moe_sort_chunks_by_index(*args, **kwargs)

# build_chunk_te_routing_map(torch.tensor([[3,2,1], [1,2,1]]), rows=12)
# out = [[0, 0, 0, 1, 1, 2, 0, 1, 1, 2, 2, 2]].T
# the pad tokens will be assigned to the last local expert so that they can be easily masked out at tail
def build_chunk_te_routing_map(
    recv_splits_by_src_local: torch.Tensor,
    *,
    rows: int,
) -> torch.Tensor:
    if recv_splits_by_src_local.ndim != 2:
        raise ValueError(
            "recv_splits_by_src_local must be rank-2 [source_rank, local_expert], "
            f"got shape={tuple(recv_splits_by_src_local.shape)}"
        )

    flat_splits_raw = recv_splits_by_src_local.reshape(-1).to(dtype=torch.long)

    num_local_experts = recv_splits_by_src_local.shape[1]


    # Keep routing-map construction tensorized on CUDA to avoid host sync.
    rows_t = flat_splits_raw.new_full((1,), rows, dtype=torch.long)
    # flat_splits_nonneg = torch.clamp_min(flat_splits_raw, 0)
    split_starts = torch.zeros_like(flat_splits_raw, dtype=torch.long)
    split_starts[1:] = torch.cumsum(flat_splits_raw[:-1], dim=0, dtype=torch.long)
    split_remaining = torch.clamp(rows_t - split_starts, min=0)
    flat_splits = torch.minimum(flat_splits_raw, split_remaining)

    pos = torch.arange(rows, device=recv_splits_by_src_local.device, dtype=torch.long)


    chunk_ends = torch.cumsum(flat_splits, dim=0)
    total_rows = chunk_ends[-1].view(1)

    valid_mask = pos < total_rows
    safe_pos = torch.where(valid_mask, pos, torch.zeros_like(pos))
    max_chunk_idx = flat_splits.numel() - 1
    chunk_ids = torch.searchsorted(chunk_ends, safe_pos, right=True).clamp_max(max_chunk_idx)
    local_expert_indices = torch.remainder(chunk_ids, num_local_experts).to(dtype=torch.int32)
    local_expert_indices = torch.where(
        valid_mask,
        local_expert_indices,
        torch.full_like(local_expert_indices, num_local_experts - 1),
    )
    return local_expert_indices.view(-1, 1)



def _moe_chunk_permute_te(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    *,
    num_out_tokens: int,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inp.ndim != 2:
        raise ValueError(f"moe_chunk_reorder_no_compile expects 2D input, got {tuple(inp.shape)}")

    permuted, row_id_map = moe_permute_no_compile(
        inp=inp,
        routing_map=routing_map,
        num_out_tokens=num_out_tokens,
        map_type="index",
    )

    if out is not None:
        out.copy_(permuted)
        permuted = out

    return permuted, row_id_map


def _moe_chunk_unpermute_te(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if inp.ndim != 2:
        raise ValueError(f"moe_chunk_reorder_no_compile expects 2D input, got {tuple(inp.shape)}")

    restored = moe_unpermute_no_compile(
        inp=inp,
        row_id_map=row_id_map,
        merging_probs=None,
        restore_shape=None,
        map_type="index",
    )

    if out is not None:
        out.copy_(restored)
        return out
    return restored


def _resolve_chunk_reorder_backend(backend: str) -> str:
    if backend == "auto":
        resolved = os.getenv("OLMO_MOE_CHUNK_REORDER_BACKEND", "cuda")
    else:
        resolved = backend
    resolved = resolved.lower()
    if resolved == "auto":
        resolved = "cuda"
    if resolved not in ("cuda", "triton", "te"):
        raise ValueError(
            "Invalid backend for moe_chunk_reorder_no_compile: "
            f"{resolved}. Expected one of cuda|triton|te (or auto -> cuda)"
        )
    return resolved


def moe_chunk_reorder_no_compile(
    inp: torch.Tensor,
    *,
    routing_map: Optional[torch.Tensor] = None,
    row_id_map: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    backend: str = "auto",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if (routing_map is None) == (row_id_map is None):
        raise ValueError(
            "Exactly one of routing_map or row_id_map must be provided to "
            "moe_chunk_reorder_no_compile"
        )

    resolved_backend = _resolve_chunk_reorder_backend(backend)

    if routing_map is not None:
        if num_out_tokens is None:
            num_out_tokens = inp.shape[0]
        routing_map_i32 = (
            routing_map
            if routing_map.dtype == torch.int32
            else routing_map.to(dtype=torch.int32)
        )

        if resolved_backend == "te":
            return _moe_chunk_permute_te(
                inp=inp,
                routing_map=routing_map_i32,
                num_out_tokens=num_out_tokens,
                out=out,
            )

        if resolved_backend in ("cuda", "triton"):
            return moe_chunk_permute(
                inp=inp,
                routing_map=routing_map_i32,
                num_out_tokens=num_out_tokens,
                out=out,
                backend=resolved_backend,  # type: ignore[arg-type]
            )
        raise RuntimeError(f"Unhandled chunk reorder backend: {resolved_backend}")

    assert row_id_map is not None
    if num_out_tokens is not None:
        raise ValueError("num_out_tokens is only valid for permute (routing_map) calls")

    row_id_map_i32 = (
        row_id_map
        if row_id_map.dtype == torch.int32
        else row_id_map.to(dtype=torch.int32)
    )

    if resolved_backend == "te":
        return _moe_chunk_unpermute_te(
            inp=inp,
            row_id_map=row_id_map_i32,
            out=out,
        )

    if resolved_backend in ("cuda", "triton"):
        return moe_chunk_unpermute(
            inp=inp,
            row_id_map=row_id_map_i32,
            num_tokens=row_id_map_i32.numel(),
            out=out,
            backend=resolved_backend,  # type: ignore[arg-type]
        )
    raise RuntimeError(f"Unhandled chunk reorder backend: {resolved_backend}")
