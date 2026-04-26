from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import nvtx
import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank
from olmo_core.kernels import symm_mem_vdev2d as symm_mem_vdev2d_kernels
from olmo_core.kernels.mxfp8_utils import (
    dequantize_rows_from_mxfp8,
    dot_gathered_rows_mxfp8_with_grad,
    quantize_rows_to_mxfp8,
)
from ...moe.utils import wait_stream_no_compile
from .routed_experts import requires_host_side_split_sizes, use_torch_grouped_mm

if TYPE_CHECKING:
    from .block import MoEFusedV2TransformerBlock


class _RowwiseCombineWeightedAutograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        expert_out: torch.Tensor,
        symm_expert_out: torch.Tensor,
        src_ranks: torch.Tensor,
        src_rows: torch.Tensor,
        probs: torch.Tensor,
        group_name: str,
        group: dist.ProcessGroup,
        nblocks: int,
    ) -> torch.Tensor:
        if expert_out.ndim != 2 or symm_expert_out.ndim != 2:
            raise RuntimeError("expert_out/symm_expert_out must be rank-2 [R, D]")
        if expert_out.shape[0] != symm_expert_out.shape[0] or expert_out.shape[1] != symm_expert_out.shape[1]:
            raise RuntimeError(
                "expert_out/symm_expert_out shape mismatch: "
                f"{tuple(expert_out.shape)} vs {tuple(symm_expert_out.shape)}"
            )
        if src_ranks.ndim != 2 or src_rows.ndim != 2:
            raise RuntimeError(
                "src_ranks/src_rows must be rank-2 [N, K], "
                f"got {tuple(src_ranks.shape)} and {tuple(src_rows.shape)}"
            )
        if src_ranks.shape != src_rows.shape:
            raise RuntimeError("src_ranks/src_rows shape mismatch")
        if probs.shape != src_ranks.shape:
            raise RuntimeError(
                "probs shape mismatch with src_ranks/src_rows: "
                f"{tuple(probs.shape)} vs {tuple(src_ranks.shape)}"
            )
        if nblocks < 0:
            raise RuntimeError(f"nblocks must be >= 0 (got {nblocks})")

        src_ranks_i64 = src_ranks if src_ranks.dtype == torch.long else src_ranks.to(dtype=torch.long)
        src_rows_i64 = src_rows if src_rows.dtype == torch.long else src_rows.to(dtype=torch.long)
        if not src_ranks_i64.is_contiguous():
            src_ranks_i64 = src_ranks_i64.contiguous()
        if not src_rows_i64.is_contiguous():
            src_rows_i64 = src_rows_i64.contiguous()

        probs_f32 = probs if probs.dtype == torch.float32 else probs.to(dtype=torch.float32)
        if not probs_f32.is_contiguous():
            probs_f32 = probs_f32.contiguous()

        input_aliases_symm_input = (
            expert_out.untyped_storage().data_ptr() == symm_expert_out.untyped_storage().data_ptr()
            and expert_out.storage_offset() == symm_expert_out.storage_offset()
            and tuple(expert_out.shape) == tuple(symm_expert_out.shape)
            and tuple(expert_out.stride()) == tuple(symm_expert_out.stride())
        )
        if not input_aliases_symm_input:
            symm_expert_out.copy_(expert_out)

        combine_out = torch.empty(
            (src_ranks_i64.shape[0], symm_expert_out.shape[1]),
            device=symm_expert_out.device,
            dtype=symm_expert_out.dtype,
        )

        need_grad_probs = ctx.needs_input_grad[4]
        if need_grad_probs:
            gathered_routes = torch.empty(
                (src_ranks_i64.shape[0], src_ranks_i64.shape[1], symm_expert_out.shape[1]),
                device=symm_expert_out.device,
                dtype=symm_expert_out.dtype,
            )
        else:
            gathered_routes = torch.empty(
                (0, 0, symm_expert_out.shape[1]),
                device=symm_expert_out.device,
                dtype=symm_expert_out.dtype,
            )

        symm_mem_vdev2d_kernels.rowwise_combine_get(
            symm_expert_out,
            combine_out,
            src_ranks_i64,
            src_rows_i64,
            group_name,
            probs=probs_f32,
            nblocks=nblocks,
            gathered_out=(gathered_routes if need_grad_probs else None),
        )

        ctx.group = group
        ctx.group_name = group_name
        ctx.nblocks = int(nblocks)
        ctx.probs_input_dtype = probs.dtype
        ctx.symm_expert_out = symm_expert_out
        ctx.save_for_backward(src_ranks_i64, src_rows_i64, probs_f32, gathered_routes)
        return combine_out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        src_ranks, src_rows, probs, gathered_routes = ctx.saved_tensors
        if grad_out.shape[0] != src_ranks.shape[0]:
            raise RuntimeError(
                "rowwise combine backward grad rows must match src_ranks rows: "
                f"{grad_out.shape[0]} vs {src_ranks.shape[0]}"
            )
        if grad_out.shape[1] != ctx.symm_expert_out.shape[1]:
            raise RuntimeError(
                "rowwise combine backward grad hidden dim must match symm_expert_out hidden dim: "
                f"{grad_out.shape[1]} vs {ctx.symm_expert_out.shape[1]}"
            )
        grad_out_contig = grad_out if grad_out.is_contiguous() else grad_out.contiguous()

        grad_probs = None
        if ctx.needs_input_grad[4]:
            grad_out_for_probs = grad_out_contig
            if grad_out_for_probs.dtype != gathered_routes.dtype:
                grad_out_for_probs = grad_out_for_probs.to(dtype=gathered_routes.dtype)
            grad_probs = torch.bmm(
                gathered_routes,
                grad_out_for_probs.unsqueeze(-1),
            ).squeeze(-1)
            if grad_probs.dtype != ctx.probs_input_dtype:
                grad_probs = grad_probs.to(dtype=ctx.probs_input_dtype)

        grad_expert_out = None
        if ctx.needs_input_grad[0]:
            symm_grad_expert_out = ctx.symm_expert_out
            # rowwise_dispatch_put only writes rows referenced by valid (src_ranks, src_rows).
            # symm_grad_expert_out is a reused shared buffer, so untouched rows can contain stale data.
            # routed_experts uses batch_size_per_expert = recv_splits_by_src_local.sum(dim=0), so backward should only consume the valid prefix/segments, not tail capacity rows.
            # Route rows are built densely for kept routes, so consumed rows should be fully overwritten.
            # symm_grad_expert_out.zero_()  <----- Likely not necessary
            symm_mem_vdev2d_kernels.rowwise_dispatch_put(
                grad_out_contig,
                symm_grad_expert_out,
                src_ranks,
                src_rows,
                ctx.group_name,
                probs=probs,
                nblocks=ctx.nblocks,
            )
            grad_expert_out = symm_grad_expert_out

        return grad_expert_out, None, None, None, grad_probs, None, None, None


class _DispatchRowwiseAutograd(torch.autograd.Function):
    @staticmethod
    # @torch.compiler.disable
    def forward(  # type: ignore[override]
        ctx,
        source_input: torch.Tensor,
        dst_ranks: torch.Tensor,
        dst_rows: torch.Tensor,
        symm_out: torch.Tensor,
        group_name: str,
        group: dist.ProcessGroup,
        nblocks: int,
    ) -> torch.Tensor:
        if source_input.ndim != 2:
            raise RuntimeError(
                f"rowwise dispatch expects source_input [N, D], got {tuple(source_input.shape)}"
            )
        if dst_ranks.shape != dst_rows.shape:
            raise RuntimeError("dst_ranks/dst_rows must have identical shapes")
        if dst_ranks.ndim != 2 or dst_ranks.shape[0] != source_input.shape[0]:
            raise RuntimeError(
                "dst_ranks/dst_rows must be [N, K] and match source_input first dim"
            )
        if symm_out.ndim != 2 or symm_out.shape[1] != source_input.shape[1]:
            raise RuntimeError(
                "symm_out must be [C, D] with matching hidden dim to source_input"
            )
        if nblocks < 0:
            raise RuntimeError(f"nblocks must be >= 0 (got {nblocks})")

        source_input_contig = source_input
        if not source_input_contig.is_contiguous():
            source_input_contig = source_input_contig.contiguous()

        dst_ranks_i64 = dst_ranks if dst_ranks.dtype == torch.long else dst_ranks.to(dtype=torch.long)
        dst_rows_i64 = dst_rows if dst_rows.dtype == torch.long else dst_rows.to(dtype=torch.long)
        if not dst_ranks_i64.is_contiguous():
            dst_ranks_i64 = dst_ranks_i64.contiguous()
        if not dst_rows_i64.is_contiguous():
            dst_rows_i64 = dst_rows_i64.contiguous()

        symm_mem_vdev2d_kernels.rowwise_dispatch_put(
            source_input_contig,
            symm_out,
            dst_ranks_i64,
            dst_rows_i64,
            group_name,
            nblocks=nblocks,
        )

        ctx.group_name = group_name
        ctx.group = group
        ctx.nblocks = int(nblocks)
        ctx.symm_out = symm_out
        ctx.save_for_backward(dst_ranks_i64, dst_rows_i64)
        return symm_out

    @staticmethod
    # @torch.compiler.disable
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        dst_ranks, dst_rows = ctx.saved_tensors
        symm_grad_out = ctx.symm_out
        grad_out_aliases = (
            grad_out.untyped_storage().data_ptr() == symm_grad_out.untyped_storage().data_ptr()
            and grad_out.storage_offset() == symm_grad_out.storage_offset()
            and tuple(grad_out.shape) == tuple(symm_grad_out.shape)
            and tuple(grad_out.stride()) == tuple(symm_grad_out.stride())
        )
        if not grad_out_aliases:
            symm_grad_out.copy_(grad_out)

        grad_input = torch.empty(
            (dst_ranks.shape[0], symm_grad_out.shape[1]),
            device=symm_grad_out.device,
            dtype=symm_grad_out.dtype,
        )
        symm_mem_vdev2d_kernels.rowwise_combine_get(
            symm_grad_out,
            grad_input,
            dst_ranks,
            dst_rows,
            ctx.group_name,
            nblocks=ctx.nblocks,
        )
        return grad_input, None, None, None, None, None, None


class _DispatchRowwiseFP8Autograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        source_input: torch.Tensor,
        dst_ranks: torch.Tensor,
        dst_rows: torch.Tensor,
        symm_out_hp: torch.Tensor,
        symm_out_q: torch.Tensor,
        symm_out_scales: torch.Tensor,
        block_size: int,
        group_name: str,
        group: dist.ProcessGroup,
        nblocks: int,
    ) -> torch.Tensor:
        if source_input.ndim != 2:
            raise RuntimeError(
                f"rowwise FP8 dispatch expects source_input [N, D], got {tuple(source_input.shape)}"
            )
        if dst_ranks.shape != dst_rows.shape:
            raise RuntimeError("dst_ranks/dst_rows must have identical shapes")
        if dst_ranks.ndim != 2 or dst_ranks.shape[0] != source_input.shape[0]:
            raise RuntimeError(
                "dst_ranks/dst_rows must be [N, K] and match source_input first dim"
            )
        if symm_out_hp.ndim != 2 or symm_out_hp.shape[1] != source_input.shape[1]:
            raise RuntimeError("symm_out_hp must be [C, D] with D matching source_input")
        if symm_out_q.shape != symm_out_hp.shape:
            raise RuntimeError(
                f"symm_out_q shape mismatch: expected {tuple(symm_out_hp.shape)}, got {tuple(symm_out_q.shape)}"
            )
        if symm_out_q.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"symm_out_q must be float8_e4m3fn, got {symm_out_q.dtype}"
            )
        if block_size <= 0 or symm_out_hp.shape[1] % block_size != 0:
            raise RuntimeError(
                f"Invalid block_size={block_size} for hidden dim {symm_out_hp.shape[1]}"
            )
        expected_scales_shape = (symm_out_hp.shape[0], symm_out_hp.shape[1] // block_size)
        if tuple(symm_out_scales.shape) != expected_scales_shape:
            raise RuntimeError(
                "symm_out_scales shape mismatch: "
                f"expected {expected_scales_shape}, got {tuple(symm_out_scales.shape)}"
            )
        if symm_out_scales.dtype != torch.float8_e8m0fnu:
            raise RuntimeError(
                f"symm_out_scales must be float8_e8m0fnu, got {symm_out_scales.dtype}"
            )
        if nblocks < 0:
            raise RuntimeError(f"nblocks must be >= 0 (got {nblocks})")

        source_input_contig = source_input if source_input.is_contiguous() else source_input.contiguous()
        dst_ranks_i64 = dst_ranks if dst_ranks.dtype == torch.long else dst_ranks.to(dtype=torch.long)
        dst_rows_i64 = dst_rows if dst_rows.dtype == torch.long else dst_rows.to(dtype=torch.long)
        if not dst_ranks_i64.is_contiguous():
            dst_ranks_i64 = dst_ranks_i64.contiguous()
        if not dst_rows_i64.is_contiguous():
            dst_rows_i64 = dst_rows_i64.contiguous()

        symm_mem_vdev2d_kernels.rowwise_dispatch_put_scaled(
            source_input_contig,
            symm_out_q,
            symm_out_scales,
            dst_ranks_i64,
            dst_rows_i64,
            group_name,
            block_size=int(block_size),
            nblocks=nblocks,
        )
        # Keep dispatch payload fully FP8 through expert compute.
        # The bf16 mirror buffer is retained for backward scratch/output only.

        ctx.group_name = group_name
        ctx.group = group
        ctx.nblocks = int(nblocks)
        ctx.block_size = int(block_size)
        ctx.symm_out_hp = symm_out_hp
        ctx.symm_out_q = symm_out_q
        ctx.symm_out_scales = symm_out_scales
        ctx.save_for_backward(dst_ranks_i64, dst_rows_i64)
        return symm_out_hp

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        dst_ranks, dst_rows = ctx.saved_tensors
        grad_out_hp = grad_out
        if grad_out_hp.dtype != ctx.symm_out_hp.dtype:
            grad_out_hp = grad_out_hp.to(dtype=ctx.symm_out_hp.dtype)
        if not grad_out_hp.is_contiguous():
            grad_out_hp = grad_out_hp.contiguous()

        quantize_rows_to_mxfp8(
            grad_out_hp,
            block_size=ctx.block_size,
            out=ctx.symm_out_q,
            scales_out=ctx.symm_out_scales,
        )

        grad_input = torch.empty(
            (dst_ranks.shape[0], grad_out_hp.shape[1]),
            device=grad_out_hp.device,
            dtype=ctx.symm_out_hp.dtype,
        )
        symm_mem_vdev2d_kernels.rowwise_combine_get_scaled(
            ctx.symm_out_q,
            ctx.symm_out_scales,
            grad_input,
            dst_ranks,
            dst_rows,
            ctx.group_name,
            block_size=ctx.block_size,
            nblocks=ctx.nblocks,
        )
        return grad_input, None, None, None, None, None, None, None, None, None


class _RowwiseCombineWeightedFP8Autograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        expert_out: torch.Tensor,
        symm_expert_out_hp: torch.Tensor,
        src_ranks: torch.Tensor,
        src_rows: torch.Tensor,
        probs: torch.Tensor,
        symm_expert_out_q: torch.Tensor,
        symm_expert_out_scales: torch.Tensor,
        block_size: int,
        group_name: str,
        group: dist.ProcessGroup,
        nblocks: int,
    ) -> torch.Tensor:
        if expert_out.ndim != 2 or symm_expert_out_hp.ndim != 2:
            raise RuntimeError("expert_out/symm_expert_out_hp must be rank-2 [R, D]")
        if tuple(expert_out.shape) != tuple(symm_expert_out_hp.shape):
            raise RuntimeError(
                "expert_out/symm_expert_out_hp shape mismatch: "
                f"{tuple(expert_out.shape)} vs {tuple(symm_expert_out_hp.shape)}"
            )
        if src_ranks.ndim != 2 or src_rows.ndim != 2:
            raise RuntimeError(
                "src_ranks/src_rows must be rank-2 [N, K], "
                f"got {tuple(src_ranks.shape)} and {tuple(src_rows.shape)}"
            )
        if src_ranks.shape != src_rows.shape:
            raise RuntimeError("src_ranks/src_rows shape mismatch")
        if probs.shape != src_ranks.shape:
            raise RuntimeError(
                f"probs shape mismatch with src_ranks/src_rows: {tuple(probs.shape)} vs {tuple(src_ranks.shape)}"
            )
        if block_size <= 0 or symm_expert_out_hp.shape[1] % block_size != 0:
            raise RuntimeError(
                f"Invalid block_size={block_size} for hidden dim {symm_expert_out_hp.shape[1]}"
            )
        if tuple(symm_expert_out_q.shape) != tuple(symm_expert_out_hp.shape):
            raise RuntimeError(
                "symm_expert_out_q shape mismatch: "
                f"expected {tuple(symm_expert_out_hp.shape)}, got {tuple(symm_expert_out_q.shape)}"
            )
        if symm_expert_out_q.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"symm_expert_out_q must be float8_e4m3fn, got {symm_expert_out_q.dtype}"
            )
        expected_scales_shape = (
            symm_expert_out_hp.shape[0],
            symm_expert_out_hp.shape[1] // block_size,
        )
        if tuple(symm_expert_out_scales.shape) != expected_scales_shape:
            raise RuntimeError(
                "symm_expert_out_scales shape mismatch: "
                f"expected {expected_scales_shape}, got {tuple(symm_expert_out_scales.shape)}"
            )
        if symm_expert_out_scales.dtype != torch.float8_e8m0fnu:
            raise RuntimeError(
                f"symm_expert_out_scales must be float8_e8m0fnu, got {symm_expert_out_scales.dtype}"
            )
        if nblocks < 0:
            raise RuntimeError(f"nblocks must be >= 0 (got {nblocks})")

        src_ranks_i64 = src_ranks if src_ranks.dtype == torch.long else src_ranks.to(dtype=torch.long)
        src_rows_i64 = src_rows if src_rows.dtype == torch.long else src_rows.to(dtype=torch.long)
        if not src_ranks_i64.is_contiguous():
            src_ranks_i64 = src_ranks_i64.contiguous()
        if not src_rows_i64.is_contiguous():
            src_rows_i64 = src_rows_i64.contiguous()

        probs_f32 = probs if probs.dtype == torch.float32 else probs.to(dtype=torch.float32)
        if not probs_f32.is_contiguous():
            probs_f32 = probs_f32.contiguous()

        expert_out_contig = expert_out if expert_out.is_contiguous() else expert_out.contiguous()
        quantize_rows_to_mxfp8(
            expert_out_contig,
            block_size=int(block_size),
            out=symm_expert_out_q,
            scales_out=symm_expert_out_scales,
        )

        combine_out = torch.empty(
            (src_ranks_i64.shape[0], symm_expert_out_hp.shape[1]),
            device=symm_expert_out_hp.device,
            dtype=symm_expert_out_hp.dtype,
        )

        need_grad_probs = ctx.needs_input_grad[4]
        if need_grad_probs:
            gathered_q_saved = torch.empty(
                (src_ranks_i64.shape[0], src_ranks_i64.shape[1], symm_expert_out_q.shape[1]),
                device=symm_expert_out_q.device,
                dtype=symm_expert_out_q.dtype,
            )
            gathered_scales_saved = torch.empty(
                (
                    src_ranks_i64.shape[0],
                    src_ranks_i64.shape[1],
                    symm_expert_out_scales.shape[1],
                ),
                device=symm_expert_out_scales.device,
                dtype=symm_expert_out_scales.dtype,
            )
        else:
            gathered_q_saved = torch.empty(
                (0, 0, symm_expert_out_q.shape[1]),
                device=symm_expert_out_q.device,
                dtype=symm_expert_out_q.dtype,
            )
            gathered_scales_saved = torch.empty(
                (0, 0, symm_expert_out_scales.shape[1]),
                device=symm_expert_out_scales.device,
                dtype=symm_expert_out_scales.dtype,
            )

        symm_mem_vdev2d_kernels.rowwise_combine_get_scaled(
            symm_expert_out_q,
            symm_expert_out_scales,
            combine_out,
            src_ranks_i64,
            src_rows_i64,
            group_name,
            probs=probs_f32,
            block_size=int(block_size),
            nblocks=nblocks,
            gathered_q_out=(gathered_q_saved if need_grad_probs else None),
            gathered_scales_out=(gathered_scales_saved if need_grad_probs else None),
        )

        ctx.group = group
        ctx.group_name = group_name
        ctx.nblocks = int(nblocks)
        ctx.block_size = int(block_size)
        ctx.probs_input_dtype = probs.dtype
        ctx.symm_expert_out_hp = symm_expert_out_hp
        ctx.symm_expert_out_q = symm_expert_out_q
        ctx.symm_expert_out_scales = symm_expert_out_scales
        ctx.save_for_backward(src_ranks_i64, src_rows_i64, probs_f32, gathered_q_saved, gathered_scales_saved)
        return combine_out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        src_ranks, src_rows, probs, gathered_q_saved, gathered_scales_saved = ctx.saved_tensors
        if grad_out.shape[0] != src_ranks.shape[0]:
            raise RuntimeError(
                "rowwise combine FP8 backward grad rows must match src_ranks rows: "
                f"{grad_out.shape[0]} vs {src_ranks.shape[0]}"
            )
        if grad_out.shape[1] != ctx.symm_expert_out_hp.shape[1]:
            raise RuntimeError(
                "rowwise combine FP8 backward grad hidden dim must match symm_expert_out hidden dim: "
                f"{grad_out.shape[1]} vs {ctx.symm_expert_out_hp.shape[1]}"
            )
        grad_out_contig = grad_out if grad_out.is_contiguous() else grad_out.contiguous()

        grad_probs = None
        valid_mask = (src_ranks >= 0) & (src_rows >= 0) & (src_rows < ctx.symm_expert_out_hp.shape[0])
        if ctx.needs_input_grad[4]:
            grad_probs = dot_gathered_rows_mxfp8_with_grad(
                gathered_q_saved,
                gathered_scales_saved,
                grad_out_contig,
                valid_mask=valid_mask,
                block_size=ctx.block_size,
                out_dtype=ctx.probs_input_dtype,
            )

        grad_expert_out = None
        if ctx.needs_input_grad[0]:
            num_rows, top_k = src_ranks.shape
            hidden = grad_out_contig.shape[1]
            weighted_flat = (
                grad_out_contig.unsqueeze(1)
                * probs.to(dtype=grad_out_contig.dtype).unsqueeze(-1)
            ).reshape(num_rows * top_k, hidden)
            if not weighted_flat.is_contiguous():
                weighted_flat = weighted_flat.contiguous()

            flat_ranks = src_ranks.reshape(-1, 1).contiguous()
            flat_rows = src_rows.reshape(-1, 1).contiguous()
            symm_mem_vdev2d_kernels.rowwise_dispatch_put_scaled(
                weighted_flat,
                ctx.symm_expert_out_q,
                ctx.symm_expert_out_scales,
                flat_ranks,
                flat_rows,
                ctx.group_name,
                block_size=ctx.block_size,
                nblocks=ctx.nblocks,
            )
            dequantize_rows_from_mxfp8(
                ctx.symm_expert_out_q,
                ctx.symm_expert_out_scales,
                block_size=ctx.block_size,
                out_dtype=ctx.symm_expert_out_hp.dtype,
                out=ctx.symm_expert_out_hp,
            )
            grad_expert_out = ctx.symm_expert_out_hp

        return grad_expert_out, None, None, None, grad_probs, None, None, None, None, None, None

@nvtx.annotate("_build_rowwise_route_maps")
def build_rowwise_route_maps(
    block: MoEFusedV2TransformerBlock,
    *,
    routing_map: torch.Tensor,
    allowed_splits: torch.Tensor,
    keep_from_src_dest_local: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-route destination rank/row maps for row-wise dispatch.
    Routes dropped by tail-capacity are encoded as -1.
    """
    self = block
    assert self.ep_pg is not None
    assert self.num_local_routed_experts is not None
    if routing_map.ndim != 2:
        raise RuntimeError(
            f"routing_map must be rank-2 [N, K], got shape={tuple(routing_map.shape)}"
        )

    num_tokens, top_k = routing_map.shape
    num_routes = num_tokens * top_k
    expert_count = self.ep_world_size * self.num_local_routed_experts

    if allowed_splits.numel() != expert_count:
        raise RuntimeError(
            "allowed_splits size mismatch: "
            f"got {allowed_splits.numel()}, expected {expert_count}"
        )
    allowed_splits_i64 = allowed_splits.to(dtype=torch.long)

    expected_keep_shape = (
        self.ep_world_size,
        self.ep_world_size,
        self.num_local_routed_experts,
    )
    if tuple(keep_from_src_dest_local.shape) != expected_keep_shape:
        raise RuntimeError(
            "keep_from_src_dest_local shape mismatch: "
            f"got {tuple(keep_from_src_dest_local.shape)}, expected {expected_keep_shape}"
        )
    keep_matrix = keep_from_src_dest_local.to(dtype=torch.long)
    if num_routes == 0:
        dst_ranks_flat = torch.full(
            (0,),
            -1,
            dtype=torch.long,
            device=routing_map.device,
        )
        dst_rows_flat = torch.full(
            (0,),
            -1,
            dtype=torch.long,
            device=routing_map.device,
        )
        return (
            dst_ranks_flat.view(num_tokens, top_k),
            dst_rows_flat.view(num_tokens, top_k),
        )

    route_experts = routing_map.reshape(-1).to(dtype=torch.long)
    valid_mask = (route_experts >= 0) & (route_experts < expert_count)
    safe_experts = torch.where(
        valid_mask,
        route_experts,
        torch.zeros_like(route_experts),
    )

    # Compute stable in-expert position for each route without dynamic-shape
    # indexing (avoids host sync from nonzero/item on CUDA tensors).
    invalid_bucket = expert_count
    bucket_ids = torch.where(
        valid_mask,
        safe_experts,
        torch.full_like(safe_experts, invalid_bucket),
    )
    sort_order = torch.argsort(bucket_ids, stable=True)
    sorted_bucket_ids = bucket_ids.index_select(0, sort_order)
    counts_per_bucket = torch.zeros(
        (expert_count + 1,),
        device=routing_map.device,
        dtype=torch.long,
    )
    counts_per_bucket.scatter_add_(
        0,
        bucket_ids,
        torch.ones_like(bucket_ids, dtype=torch.long),
    )
    starts_per_bucket = torch.cumsum(counts_per_bucket, dim=0) - counts_per_bucket
    sorted_pos = torch.arange(
        num_routes,
        device=routing_map.device,
        dtype=torch.long,
    ) - starts_per_bucket.index_select(0, sorted_bucket_ids)

    pos_in_bucket = torch.empty_like(sorted_pos)
    pos_in_bucket.scatter_(0, sort_order, sorted_pos)

    keep_limits = allowed_splits_i64.index_select(0, safe_experts)
    kept_mask = valid_mask & (pos_in_bucket < keep_limits)

    dst_rank = torch.div(
        safe_experts,
        self.num_local_routed_experts,
        rounding_mode="floor",
    )
    dst_local_expert = torch.remainder(
        safe_experts,
        self.num_local_routed_experts,
    )

    prefix_by_source = torch.cumsum(keep_matrix, dim=0) - keep_matrix
    src_rank = get_rank(self.ep_pg)
    send_base_by_dest_local = prefix_by_source[src_rank]
    recv_total_by_dest_local = keep_matrix.sum(dim=0)
    local_expert_base_by_dest = (
        torch.cumsum(recv_total_by_dest_local, dim=1) - recv_total_by_dest_local
    )

    base_rows = local_expert_base_by_dest[dst_rank, dst_local_expert]
    base_rows = base_rows + send_base_by_dest_local[dst_rank, dst_local_expert]
    dst_rows_all = base_rows + pos_in_bucket

    neg_ones = torch.full_like(dst_rank, -1)
    dst_ranks_flat = torch.where(kept_mask, dst_rank, neg_ones)
    dst_rows_flat = torch.where(kept_mask, dst_rows_all, neg_ones)
    return (
        dst_ranks_flat.view(num_tokens, top_k),
        dst_rows_flat.view(num_tokens, top_k),
    )


@nvtx.annotate("_build_rowwise_combine_2d_route_to_packed")
def build_rowwise_combine_2d_route_to_packed(
    block: MoEFusedV2TransformerBlock,
    *,
    route_to_packed: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build packed destination tensors for row-wise combine in route-order packing.
    Returns:
      - packed_dst_ranks [N*K] int64
      - packed_dst_rows [N*K] int64
      - packed_keep_mask [N*K] bool
      - num_kept_routes [] int64
      - te_row_id_map [K*N] int64
    """
    del block
    if route_to_packed.ndim != 2:
        raise RuntimeError(
            f"route_to_packed must be rank-2 [N, K], got shape={tuple(route_to_packed.shape)}"
        )
    if tuple(dst_ranks.shape) != tuple(route_to_packed.shape):
        raise RuntimeError(
            "dst_ranks shape mismatch: "
            f"got {tuple(dst_ranks.shape)}, expected {tuple(route_to_packed.shape)}"
        )
    if tuple(dst_rows.shape) != tuple(route_to_packed.shape):
        raise RuntimeError(
            "dst_rows shape mismatch: "
            f"got {tuple(dst_rows.shape)}, expected {tuple(route_to_packed.shape)}"
        )

    flat_route_to_packed = route_to_packed.reshape(-1).to(dtype=torch.long)
    flat_dst_ranks = dst_ranks.reshape(-1).to(dtype=torch.long)
    flat_dst_rows = dst_rows.reshape(-1).to(dtype=torch.long)
    flat_valid_routes = flat_route_to_packed.ge(0)
    num_routes = flat_route_to_packed.numel()

    invalid_dst = torch.full_like(flat_dst_ranks, -1)
    sentinel_index = torch.full_like(flat_route_to_packed, num_routes)
    scatter_indices = torch.where(
        flat_valid_routes,
        flat_route_to_packed,
        sentinel_index,
    )
    scatter_dst_ranks = torch.where(flat_valid_routes, flat_dst_ranks, invalid_dst)
    scatter_dst_rows = torch.where(flat_valid_routes, flat_dst_rows, invalid_dst)
    packed_dst_ranks_ext = torch.full(
        (num_routes + 1,),
        -1,
        device=flat_dst_ranks.device,
        dtype=torch.long,
    )
    packed_dst_rows_ext = torch.full_like(packed_dst_ranks_ext, -1)
    packed_dst_ranks_ext.scatter_(0, scatter_indices, scatter_dst_ranks)
    packed_dst_rows_ext.scatter_(0, scatter_indices, scatter_dst_rows)
    packed_dst_ranks = packed_dst_ranks_ext.narrow(0, 0, num_routes)
    packed_dst_rows = packed_dst_rows_ext.narrow(0, 0, num_routes)
    packed_keep_mask = packed_dst_ranks.ge(0)
    num_kept_routes = packed_keep_mask.to(dtype=torch.long).sum(dtype=torch.long)
    te_row_id_map = route_to_packed.transpose(0, 1).contiguous().reshape(-1).to(dtype=torch.long)
    return (
        packed_dst_ranks,
        packed_dst_rows,
        packed_keep_mask,
        num_kept_routes,
        te_row_id_map,
    )


def combined_forward_ep_no_sync_rowwise(
    block: MoEFusedV2TransformerBlock,
    x: torch.Tensor,
    *,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """Forward with EP no-sync using row-wise NVSHMEM dispatch/combine."""
    self = block
    assert self.routed_experts is not None
    assert self.routed_experts_router is not None
    assert self.ep_enabled
    assert self.num_local_routed_experts is not None
    assert use_torch_grouped_mm() == True, "EP no-sync implementation requires torch.grouped_mm support"
    assert not requires_host_side_split_sizes(), "EP no-sync implementation does not support host-side split size communication"
    if self.ep_no_sync_use_2d_all_to_all:
        raise RuntimeError(
            "ep_no_sync_use_2d_all_to_all=True is no longer supported: "
            "the 2D all_to_all path was removed due to correctness/performance issues."
        )

    group_name = self._get_ep_no_sync_group_name()
    B, S, D = x.shape

    block_inp = x
    del x

    attn_res_out = self._checkpointed_res_norm_attn(block_inp, **kwargs)

    kwargs.pop("max_doc_len", None)
    kwargs.pop("cu_doc_lens", None)
    moe_inp = self._prepare_moe_input(attn_res_out)

    (
        local_x_global_routed_expert_weights,
        local_x_global_routed_expert_indices,
        local_batch_size_per_global_routed_expert,
        routed_expert_router_aux_loss_info,
    ) = self.router_forward(
        router=self.routed_experts_router,
        local_x=moe_inp,
        scores_only=False,
        loss_div_factor=loss_div_factor,
    )

    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )

    with torch.cuda.stream(self.get_dense_stream()):
        if self.shared_experts_router:
            (
                local_x_global_shared_expert_weights,
                _,
                _,
                _,
            ) = self.router_forward(
                router=self.shared_experts_router,
                local_x=moe_inp,
                scores_only=True,
                loss_div_factor=loss_div_factor,
            )
        else:
            local_x_global_shared_expert_weights = None

    in_shape = moe_inp.size()
    moe_inp = moe_inp.view(-1, in_shape[-1])
    rowwise_fp8_cfg = self.rowwise_fp8
    use_rowwise_fp8 = (
        rowwise_fp8_cfg is not None
        and rowwise_fp8_cfg.enabled
        and moe_inp.device.type == "cuda"
        and self.ep_no_sync_use_rowwise_all_to_all
    )
    if use_rowwise_fp8:
        assert rowwise_fp8_cfg is not None
        if not self._rowwise_fp8_checked:
            rowwise_fp8_cfg.assert_runtime_supported()
            self._rowwise_fp8_checked = True
    else:
        rowwise_fp8_cfg = None

    num_out_tokens = local_x_global_routed_expert_indices.numel()

    with torch.no_grad():
        with nvtx.annotate("ConfigCapacity", color="green"):
            requested_splits = local_batch_size_per_global_routed_expert.to(dtype=torch.long)
            rank_capacity = self._compute_ep_no_sync_rank_capacity(num_out_tokens)
            (
                allowed_splits,
                recv_splits_by_src_local,
                _drop_token_cnt,
                keep_from_src_dest_local,
            ) = cast(
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                self._sync_tail_drop_allowed_splits_single_a2a(
                    requested_splits,
                    rank_capacity=rank_capacity,
                    return_keep_matrix=True,
                ),
            )
            dispatch_in_cap = num_out_tokens
            dispatch_out_cap = rank_capacity
            combine_in_cap = rank_capacity
            combine_out_cap = num_out_tokens
            self._accumulate_ep_no_sync_rowwise_metrics(
                drop_token_cnt=_drop_token_cnt,
                num_out_tokens=num_out_tokens,
                recv_splits_by_src_local=recv_splits_by_src_local,
                rank_capacity=rank_capacity,
            )

    buffers = self._get_ep_no_sync_buffers(
        dispatch_in_cap=dispatch_in_cap,
        dispatch_out_cap=dispatch_out_cap,
        combine_in_cap=combine_in_cap,
        combine_out_cap=combine_out_cap,
        d_model=moe_inp.shape[-1],
        dtype=moe_inp.dtype,
        device=moe_inp.device,
        need_dispatch_in=False,
        need_dispatch_meta=False,
        need_combine_meta=False,
        need_combine_out=False,
    )

    dispatch_out_q: Optional[torch.Tensor] = None
    dispatch_out_scales: Optional[torch.Tensor] = None
    combine_in_q: Optional[torch.Tensor] = None
    combine_in_scales: Optional[torch.Tensor] = None
    if use_rowwise_fp8:
        assert rowwise_fp8_cfg is not None
        if moe_inp.shape[1] % rowwise_fp8_cfg.block_size != 0:
            raise RuntimeError(
                "Rowwise FP8 requires hidden dim divisible by block_size: "
                f"hidden={moe_inp.shape[1]} block_size={rowwise_fp8_cfg.block_size}"
            )
        scale_cols = moe_inp.shape[1] // rowwise_fp8_cfg.block_size
        dispatch_out_q = self._get_or_init_ep_no_sync_symm_tensor(
            name="dispatch_out_rowwise_fp8_q",
            shape=(dispatch_out_cap, moe_inp.shape[1]),
            dtype=torch.float8_e4m3fn,
            device=moe_inp.device,
        )
        dispatch_out_scales = self._get_or_init_ep_no_sync_symm_tensor(
            name="dispatch_out_rowwise_fp8_scales",
            shape=(dispatch_out_cap, scale_cols),
            dtype=torch.float8_e8m0fnu,
            device=moe_inp.device,
        )
        combine_in_q = self._get_or_init_ep_no_sync_symm_tensor(
            name="combine_in_rowwise_fp8_q",
            shape=(combine_in_cap, moe_inp.shape[1]),
            dtype=torch.float8_e4m3fn,
            device=moe_inp.device,
        )
        combine_in_scales = self._get_or_init_ep_no_sync_symm_tensor(
            name="combine_in_rowwise_fp8_scales",
            shape=(combine_in_cap, scale_cols),
            dtype=torch.float8_e8m0fnu,
            device=moe_inp.device,
        )

    routing_map = local_x_global_routed_expert_indices.view(
        -1, self.routed_experts_router.top_k
    ).int()

    with torch.no_grad():
        padded_batch_size_per_local_expert = recv_splits_by_src_local.sum(
            dim=0,
            dtype=torch.long,
        )

    with torch.no_grad():
        dst_ranks, dst_rows = self._build_rowwise_route_maps(
            routing_map=routing_map,
            allowed_splits=allowed_splits,
            keep_from_src_dest_local=keep_from_src_dest_local,
        )
        rowwise_nblocks = self.ep_no_sync_rowwise_nblocks

    if self.shared_experts is not None:
        with torch.cuda.stream(self.get_dense_stream()):
            if use_rowwise_fp8:
                assert rowwise_fp8_cfg is not None
                shared_out_up, shared_out_gate = self._shared_experts_forward1_rowwise_fp8(
                    moe_inp,
                    use_fast_accum=rowwise_fp8_cfg.use_fast_accum,
                )
            else:
                shared_out_up, shared_out_gate = self.shared_experts.forward1(moe_inp.view(B, S, D))
    else:
        shared_out_up, shared_out_gate = None, None

    with nvtx.annotate("Rowwise Dispatch", color="green"):
        if use_rowwise_fp8:
            assert rowwise_fp8_cfg is not None
            assert dispatch_out_q is not None
            assert dispatch_out_scales is not None
            dispatch_rank_major = _DispatchRowwiseFP8Autograd.apply(
                moe_inp,
                dst_ranks,
                dst_rows,
                buffers.dispatch_out,
                dispatch_out_q,
                dispatch_out_scales,
                rowwise_fp8_cfg.block_size,
                group_name,
                self.ep_pg,
                rowwise_nblocks,
            )
        else:
            dispatch_rank_major = _DispatchRowwiseAutograd.apply(
                moe_inp,
                dst_ranks,
                dst_rows,
                buffers.dispatch_out,
                group_name,
                self.ep_pg,
                rowwise_nblocks,
            )

    dispatch_rank_major = self.routed_experts(
        dispatch_rank_major,
        padded_batch_size_per_local_expert,
        down_proj_out=(None if use_rowwise_fp8 else buffers.combine_in.detach()),
        up_proj_input_grad_out=(None if use_rowwise_fp8 else buffers.dispatch_out.detach()),
        use_rowwise_fp8=use_rowwise_fp8,
        rowwise_fp8_input_q=(dispatch_out_q if use_rowwise_fp8 else None),
        rowwise_fp8_input_scales=(dispatch_out_scales if use_rowwise_fp8 else None),
    )

    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )

    with nvtx.annotate("Rowwise Combine Merge", color="green"):
        route_probs = local_x_global_routed_expert_weights.view(
            -1, self.routed_experts_router.top_k
        )

        if use_rowwise_fp8:
            assert rowwise_fp8_cfg is not None
            assert combine_in_q is not None
            assert combine_in_scales is not None
            local_x = _RowwiseCombineWeightedFP8Autograd.apply(
                dispatch_rank_major,
                buffers.combine_in,
                dst_ranks,
                dst_rows,
                route_probs,
                combine_in_q,
                combine_in_scales,
                rowwise_fp8_cfg.block_size,
                group_name,
                self.ep_pg,
                self.ep_no_sync_rowwise_nblocks,
            )
        else:
            local_x = _RowwiseCombineWeightedAutograd.apply(
                dispatch_rank_major,
                buffers.combine_in,
                dst_ranks,
                dst_rows,
                route_probs,
                group_name,
                self.ep_pg,
                self.ep_no_sync_rowwise_nblocks,
            )

    if self.shared_experts is not None:
        assert shared_out_up is not None
        assert shared_out_gate is not None

        with torch.cuda.stream(self.get_dense_stream()):
            if use_rowwise_fp8:
                assert rowwise_fp8_cfg is not None
                shared_out = self._shared_experts_forward2_rowwise_fp8(
                    shared_out_up,
                    shared_out_gate,
                    attn_res_out.shape,
                    use_fast_accum=rowwise_fp8_cfg.use_fast_accum,
                )
            else:
                shared_out = self.shared_experts.forward2(shared_out_up, shared_out_gate, attn_res_out.shape)
            if self.shared_experts_router:
                assert local_x_global_shared_expert_weights is not None
                _, _, E_s = local_x_global_shared_expert_weights.shape
                mixed_shared_out = torch.bmm(
                    local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(B * S, 1, E_s),
                    shared_out.permute(1, 2, 0, 3).contiguous().view(B * S, E_s, D),
                ).squeeze(1).view(B, S, D)
            else:
                mixed_shared_out = shared_out.squeeze(0)
    else:
        mixed_shared_out = None

    local_x = local_x.view(in_shape)
    wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

    if self.shared_experts is not None:
        assert mixed_shared_out is not None
        mlp_out = local_x + mixed_shared_out
    else:
        mlp_out = local_x

    final_out = self._res_norm_mlp(attn_res_out, mlp_out)
    return self._attach_routed_aux_loss(final_out, routed_expert_router_aux_loss_info)
