from typing import Tuple

import torch
import torch.distributed as dist

from olmo_core.kernels import symm_mem_vdev2d as symm_mem_vdev2d_kernels
from olmo_core.kernels.mxfp8_utils import (
    dequantize_rows_from_mxfp8,
    dot_gathered_rows_mxfp8_with_grad,
    quantize_rows_to_mxfp8,
)


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
        if (
            expert_out.shape[0] != symm_expert_out.shape[0]
            or expert_out.shape[1] != symm_expert_out.shape[1]
        ):
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

        src_ranks_i64 = (
            src_ranks if src_ranks.dtype == torch.long else src_ranks.to(dtype=torch.long)
        )
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
            raise RuntimeError("dst_ranks/dst_rows must be [N, K] and match source_input first dim")
        if symm_out.ndim != 2 or symm_out.shape[1] != source_input.shape[1]:
            raise RuntimeError("symm_out must be [C, D] with matching hidden dim to source_input")
        if nblocks < 0:
            raise RuntimeError(f"nblocks must be >= 0 (got {nblocks})")

        source_input_contig = source_input
        if not source_input_contig.is_contiguous():
            source_input_contig = source_input_contig.contiguous()

        dst_ranks_i64 = (
            dst_ranks if dst_ranks.dtype == torch.long else dst_ranks.to(dtype=torch.long)
        )
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
            raise RuntimeError("dst_ranks/dst_rows must be [N, K] and match source_input first dim")
        if symm_out_hp.ndim != 2 or symm_out_hp.shape[1] != source_input.shape[1]:
            raise RuntimeError("symm_out_hp must be [C, D] with D matching source_input")
        if symm_out_q.shape != symm_out_hp.shape:
            raise RuntimeError(
                f"symm_out_q shape mismatch: expected {tuple(symm_out_hp.shape)}, got {tuple(symm_out_q.shape)}"
            )
        if symm_out_q.dtype != torch.float8_e4m3fn:
            raise RuntimeError(f"symm_out_q must be float8_e4m3fn, got {symm_out_q.dtype}")
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

        source_input_contig = (
            source_input if source_input.is_contiguous() else source_input.contiguous()
        )
        dst_ranks_i64 = (
            dst_ranks if dst_ranks.dtype == torch.long else dst_ranks.to(dtype=torch.long)
        )
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

        src_ranks_i64 = (
            src_ranks if src_ranks.dtype == torch.long else src_ranks.to(dtype=torch.long)
        )
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
        ctx.save_for_backward(
            src_ranks_i64, src_rows_i64, probs_f32, gathered_q_saved, gathered_scales_saved
        )
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
        valid_mask = (
            (src_ranks >= 0) & (src_rows >= 0) & (src_rows < ctx.symm_expert_out_hp.shape[0])
        )
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
                grad_out_contig.unsqueeze(1) * probs.to(dtype=grad_out_contig.dtype).unsqueeze(-1)
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


class _DispatchVDevAutograd(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable  # this is required if dispatch_out is shared across layers, no idea why.
    def forward(  # type: ignore[override]
        ctx,
        source_input: torch.Tensor,
        in_rank_splits: torch.Tensor,
        symm_input: torch.Tensor,
        symm_in_rank_splits: torch.Tensor,
        symm_out: torch.Tensor,
        symm_out_rank_splits_offsets: torch.Tensor,
        symm_tmp_rank_splits_offsets: torch.Tensor,
        group_name: str,
        group: dist.ProcessGroup,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source_rows = source_input.shape[0]
        if source_rows != symm_input.shape[0]:
            raise RuntimeError(
                f"dispatch input rows ({source_rows}) must equal symmetric dispatch input capacity ({symm_input.shape[0]})"
            )

        input_aliases_symm_input = (
            source_input.untyped_storage().data_ptr() == symm_input.untyped_storage().data_ptr()
            and source_input.storage_offset() == symm_input.storage_offset()
            and tuple(source_input.shape) == tuple(symm_input.shape)
            and tuple(source_input.stride()) == tuple(symm_input.stride())
        )
        if not input_aliases_symm_input:
            raise RuntimeError(
                "Not Expected: dispatch source_input should alias symm_input buffer to avoid extra copy"
            )
            symm_input.copy_(source_input)
        if in_rank_splits.dtype != torch.int64:
            symm_in_rank_splits.copy_(in_rank_splits.to(dtype=torch.int64))
        else:
            symm_in_rank_splits.copy_(in_rank_splits)

        work = dist.barrier(
            group=group,
            async_op=True,
            device_ids=[symm_input.device.index],
        )
        assert work is not None
        work.block_current_stream()

        torch.ops.symm_mem.all_to_all_vdev(
            symm_input,
            symm_out,
            symm_in_rank_splits,
            symm_out_rank_splits_offsets,
            group_name,
        )

        ctx.group = group
        ctx.group_name = group_name
        ctx.source_rows = source_input.shape[0]
        ctx.symm_input = symm_input
        ctx.symm_out = symm_out
        ctx.symm_out_rank_splits_offsets = symm_out_rank_splits_offsets
        ctx.symm_tmp_rank_splits_offsets = symm_tmp_rank_splits_offsets
        out_rank_splits_offsets = symm_out_rank_splits_offsets.clone()
        # Keep metadata directly on ctx to avoid saved_tensors lifetime issues
        # under compiled autograd.
        ctx.forward_out_rank_splits_offsets = out_rank_splits_offsets
        ctx.mark_non_differentiable(out_rank_splits_offsets)
        return symm_out, out_rank_splits_offsets

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_out_rank_splits_offsets: torch.Tensor):  # type: ignore[override]
        del grad_out_rank_splits_offsets
        forward_out_rank_splits_offsets = ctx.forward_out_rank_splits_offsets

        symm_grad_out = ctx.symm_out
        if grad_out.shape[0] != symm_grad_out.shape[0]:
            raise RuntimeError(
                f"dispatch backward grad rows ({grad_out.shape[0]}) must equal symmetric dispatch grad input capacity ({symm_grad_out.shape[0]})"
            )

        symm_grad_out_aliases_grad_out = (
            grad_out.untyped_storage().data_ptr() == symm_grad_out.untyped_storage().data_ptr()
            and grad_out.storage_offset() == symm_grad_out.storage_offset()
            and tuple(grad_out.shape) == tuple(symm_grad_out.shape)
            and tuple(grad_out.stride()) == tuple(symm_grad_out.stride())
        )
        if not symm_grad_out_aliases_grad_out:
            raise RuntimeError(
                "Not Expected: dispatch backward grad_out should alias symm_grad_out buffer to avoid extra copy"
            )

        grad_symm_input = ctx.symm_input
        # Ensure any rows not written by vdev (e.g. dropped-token tail capacity)
        # stay zero without doing a defrag/gather pass.
        grad_symm_input.zero_()
        symm_forward_out_rank_splits_offsets = ctx.symm_out_rank_splits_offsets
        symm_forward_out_rank_splits_offsets.copy_(forward_out_rank_splits_offsets)
        grad_input_rank_splits_offsets = ctx.symm_tmp_rank_splits_offsets

        work = dist.barrier(
            group=ctx.group,
            async_op=True,
            device_ids=[ctx.symm_input.device.index],
        )
        assert work is not None
        work.block_current_stream()

        torch.ops.symm_mem.all_to_all_vdev(
            symm_grad_out,
            grad_symm_input,
            symm_forward_out_rank_splits_offsets[0],
            grad_input_rank_splits_offsets,
            ctx.group_name,
        )

        # 1D vdev layout is contiguous for this path; return directly.
        if grad_symm_input.shape[0] != ctx.source_rows:
            raise RuntimeError(
                f"dispatch backward produced {grad_symm_input.shape[0]} rows, expected {ctx.source_rows}"
            )
        # grad_source_input = grad_symm_input.clone() # no need to copy
        grad_source_input = grad_symm_input
        return grad_source_input, None, None, None, None, None, None, None, None


class _CombineVDevAutograd(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable
    def forward(  # type: ignore[override]
        ctx,
        input: torch.Tensor,
        in_rank_splits: torch.Tensor,
        symm_input: torch.Tensor,
        symm_in_rank_splits: torch.Tensor,
        symm_out: torch.Tensor,
        symm_out_rank_splits_offsets: torch.Tensor,
        symm_tmp_rank_splits_offsets: torch.Tensor,
        group_name: str,
        group: dist.ProcessGroup,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_rows = input.shape[0]
        if input_rows != symm_input.shape[0]:
            raise RuntimeError(
                f"combine input rows ({input_rows}) must equal symmetric combine input capacity ({symm_input.shape[0]})"
            )

        input_aliases_symm_input = (
            input.untyped_storage().data_ptr() == symm_input.untyped_storage().data_ptr()
            and input.storage_offset() == symm_input.storage_offset()
            and tuple(input.shape) == tuple(symm_input.shape)
            and tuple(input.stride()) == tuple(symm_input.stride())
        )
        if not input_aliases_symm_input:
            symm_input.copy_(input)
        if in_rank_splits.dtype != torch.int64:
            symm_in_rank_splits.copy_(in_rank_splits.to(dtype=torch.int64))
        else:
            symm_in_rank_splits.copy_(in_rank_splits)

        work = dist.barrier(
            group=group,
            async_op=True,
            device_ids=[symm_input.device.index],
        )
        assert work is not None
        work.block_current_stream()

        torch.ops.symm_mem.all_to_all_vdev(
            symm_input,
            symm_out,
            symm_in_rank_splits,
            symm_out_rank_splits_offsets,
            group_name,
        )

        ctx.group = group
        ctx.group_name = group_name
        ctx.input_rows = input_rows
        ctx.symm_input = symm_input
        ctx.symm_in_rank_splits = symm_in_rank_splits
        ctx.symm_out = symm_out
        ctx.symm_out_rank_splits_offsets = symm_out_rank_splits_offsets
        ctx.symm_tmp_rank_splits_offsets = symm_tmp_rank_splits_offsets
        out_rank_splits_offsets = symm_out_rank_splits_offsets.clone()
        # Keep metadata directly on ctx to avoid saved_tensors lifetime issues
        # under compiled autograd.
        ctx.forward_out_rank_splits_offsets = out_rank_splits_offsets
        ctx.mark_non_differentiable(out_rank_splits_offsets)

        # the user of the output is going to be unpermute kernel, which will save combine_out for backward.
        # we need to ensure combine_out will not be overwritten, by either:
        # (1) return a new tensor if the combine_out buffer is shared
        # out = torch.empty_like(symm_out)
        # out.copy_(symm_out)
        # or
        # (2) return the symm_out buffer directly if it's not shared
        out = symm_out
        return out, out_rank_splits_offsets

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_out_rank_splits_offsets: torch.Tensor):  # type: ignore[override]
        del grad_out_rank_splits_offsets
        forward_out_rank_splits_offsets = ctx.forward_out_rank_splits_offsets

        symm_grad_out = ctx.symm_out
        if grad_out.shape[0] != symm_grad_out.shape[0]:
            raise RuntimeError(
                f"combine backward grad rows ({grad_out.shape[0]}) must equal symmetric combine grad input capacity ({symm_grad_out.shape[0]})"
            )
        symm_grad_out_aliases_grad_out = (
            grad_out.untyped_storage().data_ptr() == symm_grad_out.untyped_storage().data_ptr()
            and grad_out.storage_offset() == symm_grad_out.storage_offset()
            and tuple(grad_out.shape) == tuple(symm_grad_out.shape)
            and tuple(grad_out.stride()) == tuple(symm_grad_out.stride())
        )
        if not symm_grad_out_aliases_grad_out:
            # raise RuntimeError("Not Expected: combine backward grad_out should alias symm_grad_out buffer to avoid extra copy")
            # Shared-combine_out mode may route grad through clone() and lose aliasing.
            # Copy into the symmetric buffer in that case.
            symm_grad_out.copy_(grad_out)

        symm_grad_input = ctx.symm_input
        # Do not clear the whole capacity buffer here. Properly routed
        # downstream operations only consume rows described by split metadata,
        # so unwritten tail rows are ignored and zero-fill would add bandwidth
        # cost on the 1D path.
        symm_forward_out_rank_splits = ctx.symm_in_rank_splits
        symm_forward_out_rank_splits.copy_(forward_out_rank_splits_offsets[0])
        grad_input_rank_splits_offsets = ctx.symm_tmp_rank_splits_offsets

        work = dist.barrier(
            group=ctx.group,
            async_op=True,
            device_ids=[ctx.symm_input.device.index],
        )
        assert work is not None
        work.block_current_stream()

        torch.ops.symm_mem.all_to_all_vdev(
            symm_grad_out,
            symm_grad_input,
            symm_forward_out_rank_splits,
            grad_input_rank_splits_offsets,
            ctx.group_name,
        )

        grad_input = symm_grad_input
        # grad_input = torch.empty_like(symm_grad_input)
        # grad_input.copy_(symm_grad_input)
        return grad_input, None, None, None, None, None, None, None, None, None
