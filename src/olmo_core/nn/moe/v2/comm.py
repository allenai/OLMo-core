from typing import Tuple

import torch
import torch.distributed as dist


class _DispatchVDevAutograd(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable # this is required if dispatch_out is shared across layers, no idea why.
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
            raise RuntimeError("Not Expected: dispatch source_input should alias symm_input buffer to avoid extra copy")
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
            raise RuntimeError("Not Expected: dispatch backward grad_out should alias symm_grad_out buffer to avoid extra copy")


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
        # cost on the legacy 1D path.
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
