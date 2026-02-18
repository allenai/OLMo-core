import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union, cast

import nvtx
from textual import work
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement, Shard
from torch.utils.checkpoint import checkpoint, CheckpointFunction

import olmo_core.nn.transformer.block
try:
    import torch.distributed._symmetric_memory as _symm_mem
except ImportError:
    _symm_mem = None  # type: ignore[assignment]

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.utils import barrier, get_fs_local_rank, get_rank, get_world_size
from olmo_core.ops import moe as ops
from olmo_core.ops import attach_auxiliary_loss
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import get_or_init_stream

from ...attention import AttentionConfig, RingAttentionLoadBalancerType
from ...buffer_cache import BufferCache
from ...functional import l2_normalize
from ...layer_norm import LayerNormConfig
from ...moe import MoERouterGatingFunction
from ...moe import MoERouterConfig as MoERouterConfigV1
from ...moe.loss import MoELoadBalancingLossGranularity
from ...moe.utils import async_copy_to_cpu, wait_stream_no_compile
from .routed_experts import RoutedExperts, RoutedExpertsConfig, requires_host_side_split_sizes, use_torch_grouped_mm
from .router import MoERouterConfigV2, MoERouterV2
from .shared_experts import SharedExperts
from .shared_experts import SharedExpertsConfig

# backend: transformer_engine
from ..utils import (
    moe_unpermute_no_compile,
    moe_permute_no_compile,
    moe_sort_chunks_by_index_no_compile,
)
from olmo_core.nn.transformer.config import (
    TransformerBlockConfig,
    TransformerBlockType,
)

@dataclass
class _NoSyncSymmBuffers:
    dispatch_in: torch.Tensor
    dispatch_in_splits: torch.Tensor
    dispatch_in_rank_splits: torch.Tensor
    dispatch_out: torch.Tensor
    dispatch_splits_offsets: torch.Tensor
    dispatch_tmp_splits_offsets: torch.Tensor
    dispatch_rank_splits_offsets: torch.Tensor
    dispatch_tmp_rank_splits_offsets: torch.Tensor
    combine_in: torch.Tensor
    combine_in_rank_splits: torch.Tensor
    combine_out: torch.Tensor
    combine_splits_offsets: torch.Tensor
    combine_tmp_splits_offsets: torch.Tensor
    combine_rank_splits_offsets: torch.Tensor
    combine_tmp_rank_splits_offsets: torch.Tensor


def _build_chunk_row_plan(
    splits: torch.Tensor,
    *,
    rows: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a static-shape mapping from contiguous row positions to chunk ids/positions.
    """
    if rows < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")

    pos = torch.arange(rows, device=device, dtype=torch.long)
    if splits.numel() == 0:
        empty_mask = torch.zeros(rows, device=device, dtype=torch.bool)
        return pos.new_zeros((rows,)), pos.new_zeros((rows,)), empty_mask

    splits_long = splits.to(dtype=torch.long)
    chunk_ends = torch.cumsum(splits_long, dim=0)
    total_rows = chunk_ends[-1]

    valid_mask = pos < total_rows
    safe_pos = torch.where(valid_mask, pos, torch.zeros_like(pos))
    max_chunk_idx = splits_long.numel() - 1
    chunk_ids = torch.searchsorted(chunk_ends, safe_pos, right=True).clamp_max(max_chunk_idx)
    chunk_starts = chunk_ends - splits_long
    pos_in_chunk = safe_pos - chunk_starts.index_select(0, chunk_ids)

    safe_chunk_ids = torch.where(valid_mask, chunk_ids, torch.zeros_like(chunk_ids))
    safe_pos_in_chunk = torch.where(valid_mask, pos_in_chunk, torch.zeros_like(pos_in_chunk))
    return safe_chunk_ids, safe_pos_in_chunk, valid_mask

@nvtx.annotate("_defrag_rows_by_splits_offsets_padded", color="blue")
def _defrag_rows_by_splits_offsets_padded(
    src: torch.Tensor,
    splits_offsets: torch.Tensor,
    *,
    out_rows: int,
) -> torch.Tensor:
    """Defragment chunked rows into contiguous order with static `out_rows`."""
    splits = splits_offsets[0].to(dtype=torch.long)
    offsets = splits_offsets[1].to(dtype=torch.long)

    out = src.new_zeros((out_rows, src.shape[-1]))
    if out_rows == 0 or splits.numel() == 0:
        return out

    chunk_ids, pos_in_chunk, valid_mask = _build_chunk_row_plan(
        splits,
        rows=out_rows,
        device=src.device,
    )
    src_indices = offsets.index_select(0, chunk_ids) + pos_in_chunk
    gathered = src.index_select(0, src_indices)
    out.copy_(torch.where(valid_mask.unsqueeze(-1), gathered, torch.zeros_like(gathered)))
    return out


class _DispatchVDev2DAutograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        source_input: torch.Tensor,
        reorder_indices: torch.Tensor,
        in_splits: torch.Tensor,
        symm_input: torch.Tensor,
        symm_in_splits: torch.Tensor,
        symm_out: torch.Tensor,
        symm_out_splits_offsets: torch.Tensor,
        symm_tmp_splits_offsets: torch.Tensor,
        group_name: str,
        group: dist.ProcessGroup,
        major_align: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        packed_rows = reorder_indices.shape[0]
        if packed_rows != source_input.shape[0]:
            raise RuntimeError(
                f"dispatch reorder/input size mismatch: reorder_rows={packed_rows}, input_rows={source_input.shape[0]}"
            )
        if packed_rows != symm_input.shape[0]:
            raise RuntimeError(
                f"dispatch input rows ({packed_rows}) must equal symmetric dispatch input capacity ({symm_input.shape[0]})"
            )
        
        with nvtx.annotate("DropToken", color="green"):
            torch.index_select(source_input, 0, reorder_indices, out=symm_input)

        # symm_out.zero_()
        if in_splits.dtype != torch.int64:
            symm_in_splits.copy_(in_splits.to(dtype=torch.int64))
        else:
            symm_in_splits.copy_(in_splits)
        # symm_out_splits_offsets.zero_()

        work = dist.barrier(
            group=group,
            async_op=True,
            device_ids=[symm_input.device.index],
        )
        assert work is not None
        work.block_current_stream() 

        torch.ops.symm_mem.all_to_all_vdev_2d(
            symm_input,
            symm_out,
            symm_in_splits,
            symm_out_splits_offsets,
            group_name,
            major_align=major_align,
        )

        ctx.group = group
        ctx.group_name = group_name
        ctx.source_rows = source_input.shape[0]
        ctx.symm_input_shape = symm_input.shape
        ctx.symm_input = symm_input
        ctx.symm_out = symm_out
        ctx.symm_out_splits_offsets = symm_out_splits_offsets
        ctx.symm_tmp_splits_offsets = symm_tmp_splits_offsets
        out_splits_offsets = symm_out_splits_offsets.clone()
        ctx.save_for_backward(out_splits_offsets, reorder_indices)
        ctx.mark_non_differentiable(out_splits_offsets)
        return symm_out, out_splits_offsets

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_out_splits_offsets: torch.Tensor):  # type: ignore[override]
        del grad_out_splits_offsets
        (forward_out_splits_offsets, reorder_indices) = ctx.saved_tensors

        symm_grad_out = ctx.symm_out
        if grad_out.shape[0] != symm_grad_out.shape[0]:
            raise RuntimeError(
                f"dispatch backward grad rows ({grad_out.shape[0]}) must equal symmetric dispatch grad input capacity ({symm_grad_out.shape[0]})"
            )
        symm_grad_out.copy_(grad_out)

        grad_symm_input = ctx.symm_input
        symm_forward_out_splits_offsets = ctx.symm_out_splits_offsets
        symm_forward_out_splits_offsets.copy_(forward_out_splits_offsets)
        grad_input_splits_offsets = ctx.symm_tmp_splits_offsets
        # grad_input_splits_offsets.zero_()

        work = dist.barrier(
            group=ctx.group,
            async_op=True,
            device_ids=[ctx.symm_input.device.index],
        )
        assert work is not None
        work.block_current_stream() 

        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            symm_grad_out,
            grad_symm_input,
            symm_forward_out_splits_offsets,
            grad_input_splits_offsets,
            ctx.group_name,
        )
        grad_packed_input = _defrag_rows_by_splits_offsets_padded(
            grad_symm_input,
            grad_input_splits_offsets,
            out_rows=ctx.source_rows,
        )
        if grad_packed_input.shape[0] != ctx.source_rows:
            raise RuntimeError(
                f"dispatch backward produced {grad_packed_input.shape[0]} rows, expected {ctx.source_rows}"
            )
        grad_source_input = torch.zeros_like(grad_packed_input)
        grad_source_input.index_add_(0, reorder_indices, grad_packed_input)
        return grad_source_input, None, None, None, None, None, None, None, None, None, None


class _DispatchVDevAutograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        source_input: torch.Tensor,
        reorder_indices: torch.Tensor,
        in_rank_splits: torch.Tensor,
        symm_input: torch.Tensor,
        symm_in_rank_splits: torch.Tensor,
        symm_out: torch.Tensor,
        symm_out_rank_splits_offsets: torch.Tensor,
        symm_tmp_rank_splits_offsets: torch.Tensor,
        group_name: str,
        group: dist.ProcessGroup,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        packed_rows = reorder_indices.shape[0]
        if packed_rows != source_input.shape[0]:
            raise RuntimeError(
                f"dispatch reorder/input size mismatch: reorder_rows={packed_rows}, input_rows={source_input.shape[0]}"
            )
        if packed_rows != symm_input.shape[0]:
            raise RuntimeError(
                f"dispatch input rows ({packed_rows}) must equal symmetric dispatch input capacity ({symm_input.shape[0]})"
            )

        with nvtx.annotate("DropToken", color="green"):
            torch.index_select(source_input, 0, reorder_indices, out=symm_input)
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
        ctx.save_for_backward(out_rank_splits_offsets, reorder_indices)
        ctx.mark_non_differentiable(out_rank_splits_offsets)
        return symm_out, out_rank_splits_offsets

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_out_rank_splits_offsets: torch.Tensor):  # type: ignore[override]
        del grad_out_rank_splits_offsets
        (forward_out_rank_splits_offsets, reorder_indices) = ctx.saved_tensors

        symm_grad_out = ctx.symm_out
        if grad_out.shape[0] != symm_grad_out.shape[0]:
            raise RuntimeError(
                f"dispatch backward grad rows ({grad_out.shape[0]}) must equal symmetric dispatch grad input capacity ({symm_grad_out.shape[0]})"
            )
        symm_grad_out.copy_(grad_out)

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
        grad_source_input = torch.zeros_like(grad_symm_input)
        grad_source_input.index_add_(0, reorder_indices, grad_symm_input)
        return grad_source_input, None, None, None, None, None, None, None, None, None, None


class _CombineVDev2DOffsetAutograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        input: torch.Tensor,
        in_splits_offsets: torch.Tensor,
        symm_input: torch.Tensor,
        symm_out: torch.Tensor,
        symm_out_splits_offsets: torch.Tensor,
        symm_tmp_splits_offsets: torch.Tensor,
        symm_in_splits: torch.Tensor,
        group_name: str,
        group: dist.ProcessGroup,
        major_align: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_rows = input.shape[0]
        if input_rows != symm_input.shape[0]:
            raise RuntimeError(
                f"combine input rows ({input_rows}) must equal symmetric combine input capacity ({symm_input.shape[0]})"
            )

        symm_input.copy_(input)
        
        # torch.distributed.barrier() # NOTE: this barrier does not work
        # symm_out_splits_offsets.zero_()
        assert _symm_mem is not None, "Symmetric memory ops are not available. Make sure to use a PyTorch build with NVSHMEM support and that NVSHMEM is properly initialized."

        # option 1: NOT working (not implemented: "todo" in https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/symm_mem/NVSHMEMSymmetricMemory.cu)
        # _symm_mem.rendezvous(symm_input, group=group_name).barrier()
        # _symm_mem.rendezvous(symm_out_splits_offsets, group=group_name).barrier()
        # _symm_mem.rendezvous(in_splits_offsets, group=group_name).barrier()
        # _symm_mem.rendezvous(symm_out, group=group_name).barrier()
        # _symm_mem.rendezvous(symm_tmp_splits_offsets, group=group_name).barrier()
        
        # option 2: NOT working (barrier in NCCL stream, all2all in current stream)
        # torch.distributed.barrier(async_op=True) 

        # option 2.1: working
        work = dist.barrier(
            group=group,
            async_op=True,
            device_ids=[symm_input.device.index],
        )
        assert work is not None
        work.block_current_stream() 

        # option 3: working (consider performance implications of a full barrier here)
        # torch.distributed.barrier(async_op=False) 

        # NOTE: A barrier seems necessary, 
        # otherwise the symm_out_splits_offsets[0] (the splits) will only have non-zero values for the local rank's own expert.
        # This is a race condition between local write vs RDMA write to symm_out_splits_offsets.
        # eg, correct:
        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            symm_input,
            symm_out,
            in_splits_offsets,
            symm_out_splits_offsets,
            group_name,
        )

        ctx.group = group
        ctx.group_name = group_name
        ctx.input_rows = input_rows
        ctx.symm_input = symm_input
        ctx.symm_out = symm_out
        ctx.symm_out_splits_offsets = symm_out_splits_offsets
        ctx.symm_tmp_splits_offsets = symm_tmp_splits_offsets
        ctx.symm_in_splits = symm_in_splits
        ctx.major_align = major_align
        out_splits_offsets = symm_out_splits_offsets.clone()
        ctx.save_for_backward(out_splits_offsets)
        ctx.mark_non_differentiable(out_splits_offsets)
        return symm_out, out_splits_offsets

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_out_splits_offsets: torch.Tensor):  # type: ignore[override]
        del grad_out_splits_offsets
        (forward_out_splits_offsets,) = ctx.saved_tensors

        symm_grad_out = ctx.symm_out
        if grad_out.shape[0] != symm_grad_out.shape[0]:
            raise RuntimeError(
                f"combine backward grad rows ({grad_out.shape[0]}) must equal symmetric combine grad input capacity ({symm_grad_out.shape[0]})"
            )
        symm_grad_out.copy_(grad_out)

        grad_input = ctx.symm_input
        symm_forward_out_splits = ctx.symm_in_splits
        symm_forward_out_splits.copy_(forward_out_splits_offsets[0])
        grad_input_splits_offsets = ctx.symm_tmp_splits_offsets
        # grad_input_splits_offsets.zero_()

        work = dist.barrier(
            group=ctx.group,
            async_op=True,
            device_ids=[ctx.symm_input.device.index],
        )
        assert work is not None
        work.block_current_stream() 

        torch.ops.symm_mem.all_to_all_vdev_2d(
            symm_grad_out,
            grad_input,
            symm_forward_out_splits,
            grad_input_splits_offsets,
            ctx.group_name,
            major_align=ctx.major_align,
        )

        return grad_input[: ctx.input_rows].clone(), None, None, None, None, None, None, None, None, None


class _CombineVDevAutograd(torch.autograd.Function):
    @staticmethod
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
        ctx.save_for_backward(out_rank_splits_offsets)
        ctx.mark_non_differentiable(out_rank_splits_offsets)
        return symm_out, out_rank_splits_offsets

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_out_rank_splits_offsets: torch.Tensor):  # type: ignore[override]
        del grad_out_rank_splits_offsets
        (forward_out_rank_splits_offsets,) = ctx.saved_tensors

        symm_grad_out = ctx.symm_out
        if grad_out.shape[0] != symm_grad_out.shape[0]:
            raise RuntimeError(
                f"combine backward grad rows ({grad_out.shape[0]}) must equal symmetric combine grad input capacity ({symm_grad_out.shape[0]})"
            )
        symm_grad_out.copy_(grad_out)

        grad_input = ctx.symm_input
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
            grad_input,
            symm_forward_out_rank_splits,
            grad_input_rank_splits_offsets,
            ctx.group_name,
        )

        return grad_input[: ctx.input_rows].clone(), None, None, None, None, None, None, None, None, None


@dataclass
class MoEFusedV2TransformerBlockConfig(TransformerBlockConfig):
    
    shared_experts: Optional[SharedExpertsConfig] = None
    
    routed_experts: Optional[RoutedExpertsConfig] = None
    
    shared_experts_router: Optional[MoERouterConfigV2] = None
    
    routed_experts_router: Optional[MoERouterConfigV2] = None

    checkpoint_attn: bool = False
    checkpoint_permute_moe_unpermute: bool = False
    checkpoint_combined_ep_tbo: bool = False
    checkpoint_second_unpermute: bool = False
    ep_no_sync: bool = False
    ep_no_sync_use_2d_all_to_all: bool = False
    ep_no_sync_capacity_factor: float = 1.4
    ep_no_sync_major_align: int = 1
        
    def build(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> olmo_core.nn.transformer.block.TransformerBlockBase:
        assert self.feed_forward is None and self.feed_forward_moe is None, "MoEFusedV2TransformerBlock does not support `feed_forward` or `feed_forward_moe` (use TransformerBlockConfig instead). Set `shared_experts` and `routed_experts` instead."

        kwargs = self.as_dict(exclude_none=False, recurse=False)
        kwargs.pop("name")
        kwargs.pop("feed_forward") # from parent config
        kwargs.pop("feed_forward_moe") # from parent config
        kwargs.update(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            init_device=init_device,
            cache=cache,
        )


        if self.name == TransformerBlockType.moe_fused_v2:
            return MoEFusedV2TransformerBlock(**kwargs)
        else:
            raise NotImplementedError(self.name)


    def num_params(self, d_model: int) -> int:
        block_params = 0

        block_params += self.attention.num_params(d_model)
        if self.attention_norm is not None:
            block_params += self.attention_norm.num_params(d_model)
        if self.shared_experts is not None:
            block_params += self.shared_experts.num_params()
        if self.routed_experts is not None:
            block_params += self.routed_experts.num_params()
        if self.routed_experts_router is not None:
            block_params += self.routed_experts_router.num_params()
        if self.shared_experts_router is not None:
            block_params += self.shared_experts_router.num_params()
        if self.feed_forward_norm is not None:
            block_params += self.feed_forward_norm.num_params(d_model)

        return block_params

    def num_active_params(self, d_model: int) -> int:
        block_params = 0

        block_params += self.attention.num_params(d_model)
        if self.attention_norm is not None:
            block_params += self.attention_norm.num_params(d_model)
        if self.shared_experts is not None:
            block_params += self.shared_experts.num_params()
        if self.routed_experts is not None:
            assert self.routed_experts_router is not None, "routed_experts must have a router"
            block_params += self.routed_experts.num_active_params(self.routed_experts_router.top_k)
        if self.routed_experts_router is not None:
            block_params += self.routed_experts_router.num_params()
        if self.shared_experts_router is not None:
            block_params += self.shared_experts_router.num_params()
        if self.feed_forward_norm is not None:
            block_params += self.feed_forward_norm.num_params(d_model)

        return block_params

    def flops_per_seq(self, d_model: int, seqlen: int) -> int:
        
        flops = 0

        # attention
        flops += self.attention.flops_per_seq(d_model, seqlen)


        # router 
        # (seq_len * d_model) * (d_model * num_total_experts)
        flops += 6 * seqlen * d_model * (
            (self.routed_experts_router.num_experts if self.routed_experts_router is not None else 0)
            + (self.shared_experts_router.num_experts if self.shared_experts_router is not None else 0)
        )

        # routed experts
        # (seq_len, d_model) * (d_model, expert_hidden_size) * top_k 
        # x3 for swiglu (up, down, gate)
        # x3 for fwd+bwd
        # x2 for GEMM
        if self.routed_experts is not None:
            flops += (3 * 3 * 2) * seqlen * d_model * self.routed_experts.hidden_size *  self.routed_experts_router.top_k

        # shared experts
        # (seq_len, d_model) * (d_model, expert_hidden_size) * num_experts
        # x3 for swiglu (up, down, gate)
        # x3 for fwd+bwd
        # x2 for GEMM
        if self.shared_experts is not None:
            flops += (3 * 3 * 2) * seqlen * d_model * self.shared_experts.hidden_size * self.shared_experts.num_experts

        return flops
        

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

class MoEFusedV2TransformerBlock(olmo_core.nn.transformer.block.TransformerBlockBase):
    
    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        attention: AttentionConfig,
        attention_norm: LayerNormConfig,
        routed_experts_router: Optional[MoERouterConfigV2],
        shared_experts_router: Optional[MoERouterConfigV2],
        shared_experts: Optional[SharedExpertsConfig],
        routed_experts: Optional[RoutedExpertsConfig],
        feed_forward_norm: LayerNormConfig,
        dropout: float = 0.0,
        attention_residual_alpha: Optional[float] = None,
        feed_forward_residual_alpha: Optional[float] = None,
        checkpoint_attn = False,
        checkpoint_permute_moe_unpermute = False,
        checkpoint_combined_ep_tbo = False,
        checkpoint_second_unpermute=False,
        ep_no_sync: bool = False,
        ep_no_sync_use_2d_all_to_all: bool = True,
        ep_no_sync_capacity_factor: float = 2.0,
        ep_no_sync_major_align: int = 1,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)
        assert dropout == 0.0 or dropout is None, "MoEFusedV2TransformerBlock does not support dropout"
        self.d_model = d_model
        self.block_idx = block_idx
        
        if attention_residual_alpha is not None:
            raise OLMoConfigurationError("MoEFusedV2TransformerBlock does not support attention_residual_alpha")
        if feed_forward_residual_alpha is not None:
            raise OLMoConfigurationError("MoEFusedV2TransformerBlock does not support feed_forward_residual_alpha")

        self.routed_experts: Optional[RoutedExperts]
        self.routed_experts_router: Optional[MoERouterV2]
        self.shared_experts: Optional[SharedExperts]
        self.shared_experts_router: Optional[MoERouterV2]

        ######## START: Attention ########
        self.attention = attention.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        self.attention_norm = attention_norm.build(d_model, init_device=init_device)
        ######## END: Attention ########


        ######## START: MLP ########
        assert (routed_experts is not None) or (shared_experts is not None), "At least one of routed_experts or shared_experts must be provided"

        #### Optional: routed experts ####
        if routed_experts:
            # Routed Experts enabled
            assert routed_experts_router is not None, "Need routed_experts_router when using routed experts"
            self.routed_experts = routed_experts.build(init_device=init_device)
            self.routed_experts_router = routed_experts_router.build(init_device=init_device)
        else:
            # Routed Experts not enabled
            assert routed_experts_router is None, "Should not set routed_experts_router when not using routed experts"
            self.routed_experts = None
            self.routed_experts_router = None
        #### END: Optional: routed experts ####

        

        #### Optional: shared experts ####
        if shared_experts:
            # Shared Experts enabled
            self.shared_experts = shared_experts.build(init_device=init_device)
            # Shared Experts Router
            if shared_experts.num_experts > 1:
                # Need router if more than one experts
                assert shared_experts_router is not None, "Need shared_experts_router when using shared experts with more than one expert"
                self.shared_experts_router = shared_experts_router.build(init_device=init_device)
            else:
                assert shared_experts_router is None, "Should not set shared_experts_router when using only one shared expert"
                # No router if just one
                self.shared_experts_router = None
        else:
            # Shared Experts not enabled
            assert shared_experts_router is None, "Should not set shared_experts_router when not using shared experts"
            self.shared_experts = None
            self.shared_experts_router = None
        #### END: Optional: shared experts ####


        self.feed_forward_norm = feed_forward_norm.build(d_model, init_device=init_device)

        ######## END: MLP ########
        
        self.ep_pg = None
        self._ep_enabled = False
        self.tp_pg = None
        self._tp_enabled = False

        
        # reuse the same event so that torch.compile can see the same object id and will not break the guard.
        self._dtoh_event: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event()) # cast to make pylance happy
        self._dtoh_event_send: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event()) 
        self._dtoh_event_recv: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())

        # same for tbo1
        self._dtoh_event1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event()) 
        self._dtoh_event_send1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event()) 
        self._dtoh_event_recv1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())

        self.num_local_routed_experts: Optional[int] = self.routed_experts.num_experts if self.routed_experts else None


        self.checkpoint_attn = checkpoint_attn
        self.checkpoint_permute_moe_unpermute = checkpoint_permute_moe_unpermute
        self.checkpoint_combined_ep_tbo = checkpoint_combined_ep_tbo
        self.checkpoint_second_unpermute = checkpoint_second_unpermute
        self.ep_no_sync = ep_no_sync
        self.ep_no_sync_use_2d_all_to_all = ep_no_sync_use_2d_all_to_all
        self.ep_no_sync_capacity_factor = ep_no_sync_capacity_factor
        self.ep_no_sync_major_align = ep_no_sync_major_align
        self._ep_symm_group_name: Optional[str] = None
        self._ep_no_sync_symm_cache: Dict[str, torch.Tensor] = {}
        self._ep_no_sync_last_debug: Dict[str, torch.Tensor] = {}
        # self._ep_no_sync_forward_call_count: int = 0

        if self.ep_no_sync_capacity_factor <= 0:
            raise OLMoConfigurationError(
                f"ep_no_sync_capacity_factor must be > 0 (got {self.ep_no_sync_capacity_factor})"
            )
        if self.ep_no_sync_major_align < 1:
            raise OLMoConfigurationError(
                f"ep_no_sync_major_align must be >= 1 (got {self.ep_no_sync_major_align})"
            )



    def purge_cuda_events(self):
        # set all events to None (so that the model can be deepcopied)
        self._dtoh_event = None # type: ignore[assignment]
        self._dtoh_event_send = None # type: ignore[assignment]
        self._dtoh_event_recv = None # type: ignore[assignment]
        self._before_rev_all2all_event = None # type: ignore[assignment]

        self._dtoh_event1 = None # type: ignore[assignment]
        self._dtoh_event_send1 = None # type: ignore[assignment]
        self._dtoh_event_recv1 = None # type: ignore[assignment]
        self._before_rev_all2all_event1 = None # type: ignore[assignment]

    def install_cuda_events(self):
        self._dtoh_event = cast(torch.cuda.Event, torch.cuda.Event()) 
        self._dtoh_event_send = cast(torch.cuda.Event, torch.cuda.Event()) 
        self._dtoh_event_recv = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event = cast(torch.cuda.Event, torch.cuda.Event())

        self._dtoh_event1 = cast(torch.cuda.Event, torch.cuda.Event()) 
        self._dtoh_event_send1 = cast(torch.cuda.Event, torch.cuda.Event()) 
        self._dtoh_event_recv1 = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event1 = cast(torch.cuda.Event, torch.cuda.Event())

    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        # compute shared and routed experts metrics
        # metrics_shared = self.shared_experts.compute_metrics(reset=reset)
        if self.routed_experts_router:
            metrics_routed = self.routed_experts_router.compute_metrics(reset=reset)
        else:
            metrics_routed = {}
        # metrics = {
        #     "shared": metrics_shared,
        #     "routed": metrics_routed,
        # }
        return metrics_routed

    def reset_metrics(self):
        # if self.shared_experts_router:
        #     self.shared_experts_router.reset_metrics()
        if self.routed_experts_router:
            self.routed_experts_router.reset_metrics()


    @property
    def is_moe(self) -> bool:
        return True

    @property
    def ep_enabled(self) -> bool:
        return self._ep_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    def get_dense_stream(self, for_x1=False) -> torch.cuda.Stream:
        if for_x1: # not used for now
            return get_or_init_stream(id='dense_x1', priority=20)
        else:
            return get_or_init_stream(id='dense', priority=20)

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.routed_experts:
            if self.ep_enabled:
                if self.ep_no_sync:
                    return self.combined_forward_ep_no_sync(
                        x, loss_div_factor=loss_div_factor, **kwargs
                    )
                return self.combined_forward_ep(x, loss_div_factor=loss_div_factor, **kwargs)
            else:
                return self.combined_forward_no_ep(x, loss_div_factor=loss_div_factor, **kwargs)
        else:
            # only shared_experts
            return self.combined_forward_shared_only(x, loss_div_factor=loss_div_factor, **kwargs)

    def apply_pp(self, pp_mesh: DeviceMesh):
        pass # nothing to do

    def _ensure_ep_no_sync_symm_backend(self):
        if _symm_mem is None:
            raise RuntimeError(
                "EP no-sync requires torch.distributed._symmetric_memory, but it is unavailable"
            )

        if not torch.cuda.is_available():
            raise RuntimeError("EP no-sync requires CUDA")

        device = torch.device("cuda", torch.cuda.current_device())
        current_backend = _symm_mem.get_backend(device)
        if current_backend is not None and current_backend.upper() == "NVSHMEM":
            return

        if not _symm_mem.is_nvshmem_available():
            raise RuntimeError(
                "EP no-sync requires NVSHMEM-backed symmetric memory for "
                "all_to_all_vdev/all_to_all_vdev_2d, but NVSHMEM is not available in this "
                "PyTorch build/environment."
            )

        try:
            _symm_mem.set_backend("NVSHMEM")
        except Exception as e:
            try:
                backend_after = _symm_mem.get_backend(device)
            except Exception:
                backend_after = None
            raise RuntimeError(
                "EP no-sync requires NVSHMEM-backed symmetric memory for "
                "all_to_all_vdev/all_to_all_vdev_2d. Failed to switch backend to NVSHMEM "
                f"(current={current_backend}, after_error={backend_after}): {e}. "
                "Call torch.distributed._symmetric_memory.set_backend('NVSHMEM') "
                "before any symmetric-memory allocations."
            ) from e

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        assert self.routed_experts is not None, "ep can only be applied when routed_experts is enabled"
        ep_dp_mesh = ep_mesh['ep_dp']
        ep_mp_mesh = ep_mesh['ep_mp']
        self.ep_mesh = ep_mesh
        self.routed_experts.apply_ep(
            ep_mesh
        )
        self.num_local_routed_experts = self.routed_experts.num_local_experts
        self._ep_enabled = True
        self.ep_pg = ep_mp_mesh.get_group()

        if self.ep_no_sync:
            if _symm_mem is None:
                raise RuntimeError(
                    "EP no-sync requires torch.distributed._symmetric_memory, but it is unavailable"
                )
            self._ensure_ep_no_sync_symm_backend()
            assert self.ep_pg is not None
            group_name = self.ep_pg.group_name
            assert dist.group.WORLD is not None, "torch.distributed.group.WORLD must be initialized for EP no-sync to work"
            world_group_name = dist.group.WORLD.group_name 
            try:
                # NVSHMEM-backed symm_mem.empty() allocates through the default group path.
                # Ensure WORLD group info is initialized so allocations do not fail before rendezvous.
                _symm_mem.enable_symm_mem_for_group(world_group_name) # _symm_mem.empty() resolves group info for WORLD, even when rendezvous is ep_pg, don't know why. symm API is unstable for now in torch 2.10
                _symm_mem.enable_symm_mem_for_group(group_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to enable symmetric memory for WORLD='{world_group_name}' "
                    f"and EP group '{group_name}' "
                    f"(block={self.block_idx}, rank={get_rank(self.ep_pg)}): {e}"
                ) from e
            self._ep_symm_group_name = group_name
            self._ep_no_sync_symm_cache.clear()

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        raise NotImplementedError("TP is not supported in MoEFusedV1TransformerBlock")

    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        raise NotImplementedError("CP is not supported in MoEFusedV1TransformerBlock")
        self.attention.apply_cp(cp_mesh, load_balancer)
        self.shared_experts.apply_cp(cp_mesh)
        self.routed_experts.apply_cp(cp_mesh)

    def apply_fsdp(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("FSDP is not supported in MoEFusedV2TransformerBlock")

    def apply_compile(self):
        self.compile(fullgraph=False)

        # NOTE: the tbo might be called by the outer model directly (by block.combined_forward_ep_tbo(x, ...) instead of block(x, ...)), so need to compile it here as well
        # self.combined_forward_ep_tbo = torch.compile(self.combined_forward_ep_tbo) 
        self._res_norm_attn = torch.compile(self._res_norm_attn)
        self._routed_experts_unpermute = torch.compile(self._routed_experts_unpermute)


    @property
    def ep_world_size(self) -> int:
        if self.ep_pg is not None:
            return get_world_size(self.ep_pg)
        else:
            return 1
        
    def router_forward(
        self,
        router: MoERouterV2,
        local_x: torch.Tensor,
        scores_only: bool,
        loss_div_factor: Optional[Union[torch.Tensor, float]],
    ):
        return router(
            local_x, 
            scores_only,
            loss_div_factor=loss_div_factor # scalar
        )

    def _get_ep_no_sync_group_name(self) -> str:
        if not self.ep_no_sync:
            raise RuntimeError("EP no-sync is not enabled for this block")
        if self._ep_symm_group_name is None:
            raise RuntimeError(
                f"EP no-sync group is not initialized (block={self.block_idx}, ep_enabled={self.ep_enabled})"
            )
        return self._ep_symm_group_name

    def _get_or_init_ep_no_sync_symm_tensor(
        self,
        *,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if _symm_mem is None:
            raise RuntimeError("EP no-sync requires torch.distributed._symmetric_memory")
        if self.ep_pg is None:
            raise RuntimeError("EP process group is not initialized")

        cached = self._ep_no_sync_symm_cache.get(name)
        needs_realloc = (
            cached is None
            or tuple(cached.shape) != tuple(shape)
            or cached.dtype != dtype
            or cached.device != device
        )
        if needs_realloc:
            try:
                symm_tensor = _symm_mem.empty(shape, dtype=dtype, device=device)
                _symm_mem.rendezvous(symm_tensor, group=self.ep_pg)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to allocate/rendezvous symmetric tensor '{name}' with shape={shape}, "
                    f"dtype={dtype}, device={device}, block={self.block_idx}, rank={get_rank(self.ep_pg)}: {e}"
                ) from e
            self._ep_no_sync_symm_cache[name] = symm_tensor

        return self._ep_no_sync_symm_cache[name]

    def _get_ep_no_sync_buffers(
        self,
        *,
        dispatch_in_cap: int,
        dispatch_out_cap: int,
        combine_out_cap: int,
        d_model: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _NoSyncSymmBuffers:
        assert self.routed_experts_router is not None

        num_experts = self.routed_experts_router.num_experts
        ep_world_size = self.ep_world_size
        dispatch_in = self._get_or_init_ep_no_sync_symm_tensor(
            name="dispatch_in",
            shape=(dispatch_in_cap, d_model),
            dtype=dtype,
            device=device,
        )
        dispatch_in_splits = self._get_or_init_ep_no_sync_symm_tensor(
            name="dispatch_in_splits",
            shape=(num_experts,),
            dtype=torch.int64,
            device=device,
        )
        dispatch_in_rank_splits = self._get_or_init_ep_no_sync_symm_tensor(
            name="dispatch_in_rank_splits",
            shape=(ep_world_size,),
            dtype=torch.int64,
            device=device,
        )
        dispatch_out = self._get_or_init_ep_no_sync_symm_tensor(
            name="dispatch_out",
            shape=(dispatch_out_cap, d_model),
            dtype=dtype,
            device=device,
        )
        dispatch_splits_offsets = self._get_or_init_ep_no_sync_symm_tensor(
            name="dispatch_splits_offsets",
            shape=(2, num_experts),
            dtype=torch.int64,
            device=device,
        )
        dispatch_tmp_splits_offsets = self._get_or_init_ep_no_sync_symm_tensor(
            name="dispatch_tmp_splits_offsets",
            shape=(2, num_experts),
            dtype=torch.int64,
            device=device,
        )
        dispatch_rank_splits_offsets = self._get_or_init_ep_no_sync_symm_tensor(
            name="dispatch_rank_splits_offsets",
            shape=(2, ep_world_size),
            dtype=torch.int64,
            device=device,
        )
        dispatch_tmp_rank_splits_offsets = self._get_or_init_ep_no_sync_symm_tensor(
            name="dispatch_tmp_rank_splits_offsets",
            shape=(2, ep_world_size),
            dtype=torch.int64,
            device=device,
        )
        combine_in = self._get_or_init_ep_no_sync_symm_tensor(
            name="combine_in",
            shape=(dispatch_out_cap, d_model),
            dtype=dtype,
            device=device,
        )
        combine_in_rank_splits = self._get_or_init_ep_no_sync_symm_tensor(
            name="combine_in_rank_splits",
            shape=(ep_world_size,),
            dtype=torch.int64,
            device=device,
        )
        combine_out = self._get_or_init_ep_no_sync_symm_tensor(
            name="combine_out",
            shape=(combine_out_cap, d_model),
            dtype=dtype,
            device=device,
        )
        combine_splits_offsets = self._get_or_init_ep_no_sync_symm_tensor(
            name="combine_splits_offsets",
            shape=(2, num_experts),
            dtype=torch.int64,
            device=device,
        )
        combine_tmp_splits_offsets = self._get_or_init_ep_no_sync_symm_tensor(
            name="combine_tmp_splits_offsets",
            shape=(2, num_experts),
            dtype=torch.int64,
            device=device,
        )
        combine_rank_splits_offsets = self._get_or_init_ep_no_sync_symm_tensor(
            name="combine_rank_splits_offsets",
            shape=(2, ep_world_size),
            dtype=torch.int64,
            device=device,
        )
        combine_tmp_rank_splits_offsets = self._get_or_init_ep_no_sync_symm_tensor(
            name="combine_tmp_rank_splits_offsets",
            shape=(2, ep_world_size),
            dtype=torch.int64,
            device=device,
        )
        return _NoSyncSymmBuffers(
            dispatch_in=dispatch_in,
            dispatch_in_splits=dispatch_in_splits,
            dispatch_in_rank_splits=dispatch_in_rank_splits,
            dispatch_out=dispatch_out,
            dispatch_splits_offsets=dispatch_splits_offsets,
            dispatch_tmp_splits_offsets=dispatch_tmp_splits_offsets,
            dispatch_rank_splits_offsets=dispatch_rank_splits_offsets,
            dispatch_tmp_rank_splits_offsets=dispatch_tmp_rank_splits_offsets,
            combine_in=combine_in,
            combine_in_rank_splits=combine_in_rank_splits,
            combine_out=combine_out,
            combine_splits_offsets=combine_splits_offsets,
            combine_tmp_splits_offsets=combine_tmp_splits_offsets,
            combine_rank_splits_offsets=combine_rank_splits_offsets,
            combine_tmp_rank_splits_offsets=combine_tmp_rank_splits_offsets,
        )

    def _build_padded_local_expert_batch_sizes_from_layout(
        self,
        *,
        splits: torch.Tensor,
        offsets: torch.Tensor,
        total_rows: int,
    ) -> torch.Tensor:
        """
        Build per-local-expert padded group sizes from chunked dispatch layout.
        `splits`/`offsets` are local-expert-major then source-rank.
        """
        assert self.num_local_routed_experts is not None
        expected_chunks = self.num_local_routed_experts * self.ep_world_size
        if splits.numel() != expected_chunks or offsets.numel() != expected_chunks:
            raise RuntimeError(
                "dispatch layout metadata shape mismatch: "
                f"splits={splits.numel()}, offsets={offsets.numel()}, expected={expected_chunks}"
            )

        splits_long = splits.to(dtype=torch.long)
        offsets_long = offsets.to(dtype=torch.long)
        offsets_2d = offsets_long.view(self.num_local_routed_experts, self.ep_world_size)
        expert_starts = offsets_2d[:, 0].clone()
        # grouped_mm groups always start at row 0; absorb any leading gap into expert 0.
        expert_starts[0] = 0
        expert_ends = torch.empty_like(expert_starts)
        if self.num_local_routed_experts > 1:
            expert_ends[:-1] = expert_starts[1:]
        # Exclude any trailing symmetric-buffer capacity tail that is not referenced by
        # dispatch metadata, while still including in-group/inter-group alignment pads.
        chunk_ends = offsets_long + splits_long
        used_rows = (chunk_ends * (splits_long > 0).to(dtype=torch.long)).max()
        if total_rows >= 0:
            used_rows = torch.clamp_max(used_rows, int(total_rows))
        expert_ends[-1] = used_rows
        padded_sizes = expert_ends - expert_starts
        # Guard against malformed metadata under backend issues.
        return torch.clamp_min(padded_sizes, 0)

    def _compute_ep_no_sync_rank_capacity(self, num_out_tokens: int) -> int:
        # `num_out_tokens` is the local routed-token count before EP dispatch.
        # Under balanced routing, the average received tokens per EP rank is this
        # same value (not divided by ep_world_size).
        return max(
            1,
            int(
                math.ceil(
                    self.ep_no_sync_capacity_factor * float(num_out_tokens)
                )
            ),
        )

    def _build_tail_keep_quota(
        self,
        recv_counts_per_src_local_expert: torch.Tensor,
        rank_capacity: int,
    ) -> torch.Tensor:
        """
        Build per-source keep quotas on destination rank.
        Order is local-expert-major then source-rank.
        """
        counts = recv_counts_per_src_local_expert.to(dtype=torch.long)
        # shape: (num_local_experts, ep_world_size)
        counts_flat = counts.transpose(0, 1).reshape(-1)
        cumsum_counts = torch.cumsum(counts_flat, dim=0)
        kept_cumsum = torch.clamp(cumsum_counts, max=rank_capacity)
        prev = torch.cat([torch.zeros(1, device=counts.device, dtype=torch.long), kept_cumsum[:-1]])
        kept_flat = kept_cumsum - prev
        kept = kept_flat.view(self.num_local_routed_experts, self.ep_world_size).transpose(0, 1)
        return kept

    @nvtx.annotate("SyncTokenCount", color="green")
    def _sync_tail_drop_allowed_splits_single_a2a(
        self,
        requested_splits: torch.Tensor,
        *,
        rank_capacity: int,
        return_recv_splits_by_src_local: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Tail-drop keep-split sync with a single all_to_all_single.
        Each EP rank receives every rank's requested splits and capacity, then
        computes the same keep policy locally.
        """
        assert self.num_local_routed_experts is not None
        requested = requested_splits.to(dtype=torch.long)
        expected_splits = self.ep_world_size * self.num_local_routed_experts
        if requested.numel() != expected_splits:
            raise RuntimeError(
                "requested_splits size mismatch: "
                f"got {requested.numel()}, expected {expected_splits}"
            )

        local_capacity = torch.tensor(
            [rank_capacity],
            device=requested.device,
            dtype=torch.long,
        )
        payload = torch.cat((requested, local_capacity), dim=0)
        payload_replicated = payload.repeat(self.ep_world_size)
        gathered_payload = torch.empty_like(payload_replicated)
        dist.all_to_all_single(
            gathered_payload,
            payload_replicated,
            group=self.ep_pg,
        )

        payload_width = payload.numel()
        gathered_payload_2d = gathered_payload.view(self.ep_world_size, payload_width)
        global_requested = gathered_payload_2d[:, :-1].view(
            self.ep_world_size, self.ep_world_size, self.num_local_routed_experts
        )
        global_capacity = gathered_payload_2d[:, -1]

        # Flatten in local-expert-major then source-rank order for each destination rank.
        counts_flat = global_requested.permute(2, 0, 1).reshape(-1, self.ep_world_size)
        cumsum_counts = torch.cumsum(counts_flat, dim=0)
        kept_cumsum = torch.minimum(cumsum_counts, global_capacity.view(1, -1))
        prev = torch.cat(
            [
                torch.zeros((1, self.ep_world_size), device=requested.device, dtype=torch.long),
                kept_cumsum[:-1],
            ],
            dim=0,
        )
        kept_flat = kept_cumsum - prev
        keep_from_src_dest_local = kept_flat.view(
            self.num_local_routed_experts, self.ep_world_size, self.ep_world_size
        ).permute(1, 2, 0)

        local_rank = get_rank(self.ep_pg)
        allowed_splits = keep_from_src_dest_local[local_rank].reshape(-1)
        allowed_splits = torch.minimum(allowed_splits, requested)
        if return_recv_splits_by_src_local:
            # shape: (source_rank, local_expert)
            recv_splits_by_src_local = keep_from_src_dest_local[:, local_rank, :]
            return allowed_splits, recv_splits_by_src_local
        return allowed_splits

    @nvtx.annotate('_build_keep_reorder')
    def _build_keep_reorder(
        self,
        requested_splits: torch.Tensor,
        keep_splits: torch.Tensor,
        num_out_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build a static-shape reorder map that moves kept tokens to the front while
        preserving within-group order. Returns:
          - reorder indices (original -> packed order),
          - inverse reorder indices (packed -> original order),
          - keep mask in packed order.
        """
        requested = requested_splits.to(dtype=torch.long)
        keep = keep_splits.to(dtype=torch.long)
        token_ids = torch.arange(num_out_tokens, device=keep.device, dtype=torch.long)
        if requested.numel() == 0:
            keep_mask = torch.zeros(num_out_tokens, device=keep.device, dtype=torch.bool)
            return token_ids, token_ids, keep_mask

        requested_ends = torch.cumsum(requested, dim=0)
        total_requested = requested_ends[-1]
        in_range = token_ids < total_requested
        safe_token_ids = torch.where(in_range, token_ids, torch.zeros_like(token_ids))

        max_expert_idx = requested.numel() - 1
        expert_ids = torch.searchsorted(requested_ends, safe_token_ids, right=True).clamp_max(max_expert_idx)
        starts = requested_ends - requested
        pos_in_chunk = safe_token_ids - starts.index_select(0, expert_ids)
        keep_mask = in_range & (pos_in_chunk < keep.index_select(0, expert_ids))

        # Stable partition in O(N): keep rows first, then dropped rows.
        keep_i64 = keep_mask.to(dtype=torch.long)
        drop_i64 = (~keep_mask).to(dtype=torch.long)
        keep_rank = torch.cumsum(keep_i64, dim=0) - 1
        drop_rank = torch.cumsum(drop_i64, dim=0) - 1
        num_kept = keep_i64.sum(dtype=torch.long)
        packed_pos = torch.where(keep_mask, keep_rank, num_kept + drop_rank)

        reorder_indices = torch.empty_like(token_ids)
        reorder_indices.scatter_(0, packed_pos, token_ids)

        inverse_reorder_indices = torch.empty_like(reorder_indices)
        inverse_reorder_indices.scatter_(0, reorder_indices, token_ids)
        packed_keep_mask = keep_mask.index_select(0, reorder_indices)
        return reorder_indices, inverse_reorder_indices, packed_keep_mask

    @nvtx.annotate('_restore_local_from_chunked_combine_out')
    def _restore_local_from_chunked_combine_out(
        self,
        combine_out: torch.Tensor,
        *,
        combine_splits_offsets: torch.Tensor,
        local_inverse_reorder_indices: torch.Tensor,
        packed_keep_mask: torch.Tensor,
        out_rows: int,
    ) -> torch.Tensor:
        """
        Convert chunked combine output directly to original local packed-token order.
        This fuses:
          1) gather-by-(splits,offsets) into packed order,
          2) dropped-row zeroing,
          3) inverse local reorder.
        """
        if out_rows == 0:
            return combine_out.new_zeros((0, combine_out.shape[-1]))

        splits = combine_splits_offsets[0].to(dtype=torch.long)
        offsets = combine_splits_offsets[1].to(dtype=torch.long)
        chunk_ids, pos_in_chunk, valid_mask = _build_chunk_row_plan(
            splits,
            rows=out_rows,
            device=combine_out.device,
        )
        packed_src_indices = offsets.index_select(0, chunk_ids) + pos_in_chunk

        restored_src_indices = packed_src_indices.index_select(0, local_inverse_reorder_indices)
        restored_valid_mask = valid_mask.index_select(0, local_inverse_reorder_indices)
        restored_keep_mask = packed_keep_mask.index_select(0, local_inverse_reorder_indices)
        restored_mask = restored_valid_mask & restored_keep_mask

        restored = combine_out.index_select(0, restored_src_indices)
        return torch.where(restored_mask.unsqueeze(-1), restored, torch.zeros_like(restored))



    def combined_forward_shared_only(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Reserved for no-routed-experts case (only shared experts), equivalent to a dense model"""
        assert self.routed_experts is None
        assert self.routed_experts_router is None
        assert self.shared_experts is not None
        raise NotImplementedError("combined_forward_shared_only is not implemented")


    @torch.compiler.disable
    def sync_dtoh_event(self):
        assert self._dtoh_event is not None
        dtoh_event = cast(torch.cuda.Event, self._dtoh_event)
        dtoh_event.synchronize()

    def combined_forward_no_ep(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function without EP"""
        assert self.routed_experts is not None
        assert self.routed_experts_router is not None
    
        B, S, D = x.shape

        # rename "x" to avoid confusion
        block_inp = x
        del x

        # attention 
        # + attention norm
        # + residual connection
        attn_res_out: torch.Tensor = block_inp + self.attention_norm(self.attention(block_inp, **kwargs))
        # remove attention kwargs
        kwargs.pop("max_doc_len", None)
        kwargs.pop("cu_doc_lens", None)


        # routed expert router
        (
            local_x_global_routed_expert_weights, # (B, S, top_k)
            local_x_global_routed_expert_indices, # (B, S, top_k)
            local_batch_size_per_global_routed_expert, # (num_experts, )
            routed_expert_router_aux_loss_info # tuple
        ) = self.router_forward(
            router=self.routed_experts_router,
            local_x=attn_res_out, 
            scores_only=False,
            loss_div_factor=loss_div_factor # scalar
        )

        # DtoH token count communication
        # should start DtoH as immediately after the results are available on GPU
        if requires_host_side_split_sizes():
            local_batch_size_per_global_routed_expert_cpu, copy_stream, dtoh_event = async_copy_to_cpu(local_batch_size_per_global_routed_expert, event=self._dtoh_event)  
        else:
            # for type checking, not used when host side split sizes are not required
            dtoh_event = None 
            local_batch_size_per_global_routed_expert_cpu = None
        
        # shared expert router
        if self.shared_experts_router:
            (
                local_x_global_shared_expert_weights, # (B, S, E_shared)
                _, 
                _, 
                _ 
            ) = self.router_forward(
                router=self.shared_experts_router,
                local_x=attn_res_out, 
                scores_only=True,  # only need scores for shared experts
                loss_div_factor=loss_div_factor # scalar
            )
        else:
            local_x_global_shared_expert_weights = None
        
        
        moe_inp = attn_res_out

        in_shape = moe_inp.size()

        mixed_shared_out = None
        if self.shared_experts is not None:
            # the shared experts (executed on the dense stream) need to wait for `attn_res_out` and `local_x_global_shared_expert_weights` (on the main stream) to be complete
            wait_stream_no_compile(
                this_stream=self.get_dense_stream(),
                other_stream=torch.cuda.current_stream()
            )


            # overlap compute while waiting for the copy to CPU to finish
            with torch.cuda.stream(self.get_dense_stream()):
                shared_out = self.shared_experts(moe_inp) # (E_shared, B, S, D)
                if self.shared_experts.num_experts == 1:
                    mixed_shared_out = shared_out.squeeze(0)
                else:
                    assert local_x_global_shared_expert_weights is not None
                    # weighted sum of the shared experts by router weights
                    # local_x_global_shared_expert_weights -> (B, S, E_shared)
                    # shared_out -> (E_shared, B, S, D)
                    _, _, E_s = local_x_global_shared_expert_weights.shape
                    local_x_global_shared_expert_weights.shape
                    mixed_shared_out = torch.bmm(
                        local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(B*S, 1, E_s),            # (BS, 1, E), 
                        shared_out.permute(1, 2, 0, 3).contiguous().view(B*S, E_s, D)              # (BS, E, D)
                    ).squeeze(1).view(B, S, D)
                
        
        moe_inp = moe_inp.view(-1, in_shape[-1])  # (B*S, D)



        routing_map = local_x_global_routed_expert_indices.view(-1, self.routed_experts_router.top_k).int()
        num_out_tokens = routing_map.size(0) * self.routed_experts_router.top_k # dropless
        hidden_shape_before_permute = moe_inp.shape

        # step 2: permute the input tokens
        with nvtx.annotate("Permute", color='green'):
            permutated_input_tokens, reversed_input_permutation_mapping = moe_permute_no_compile(
                inp=moe_inp, 
                routing_map=routing_map, 
                num_out_tokens=num_out_tokens, 
                map_type='index'
            )

        torch._dynamo.mark_dynamic(permutated_input_tokens, 0)

        # step 3: MLP
        ####################################
        if requires_host_side_split_sizes():
            assert dtoh_event is not None 
            dtoh_event = cast(torch.cuda.Event, dtoh_event)
            dtoh_event.synchronize()
            mlp_x = self.routed_experts(permutated_input_tokens, local_batch_size_per_global_routed_expert_cpu)
        else:
            mlp_x = self.routed_experts(permutated_input_tokens, local_batch_size_per_global_routed_expert)
        ####################################


        # step 4: unpermutate the output tokens
        with nvtx.annotate("Unpermute", color='green'):
            unpermutated_x: torch.Tensor = moe_unpermute_no_compile(
                inp=mlp_x,
                row_id_map=reversed_input_permutation_mapping,
                restore_shape=hidden_shape_before_permute,
                map_type='index',
                merging_probs=local_x_global_routed_expert_weights.view(-1, self.routed_experts_router.top_k)
            ) 
            
        x_moe = unpermutated_x.view(in_shape)

        # need to use `mixed_shared_out`
        wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())


        if self.shared_experts is not None:
            assert mixed_shared_out is not None
            # # weighted sum of the shared experts and routed experts
            # shared_width = self.shared_experts.num_experts * self.shared_experts.hidden_size
            # routed_active_width = self.routed_experts_router.top_k * self.routed_experts.hidden_size
            # total_width = shared_width + routed_active_width
            # shared_out_factor = shared_width / total_width
            # routed_out_factor = routed_active_width / total_width
            # mlp_out = self.merge_shared_and_routed_out(
            #     shared_out=mixed_shared_out,
            #     shared_factor=shared_out_factor,
            #     routed_out=x_moe,
            #     routed_factor=routed_out_factor
            # )
            mlp_out = x_moe + mixed_shared_out
        else:
            mlp_out = x_moe # only routed experts

        final_out = attn_res_out + self.feed_forward_norm(mlp_out)

        #######################

        # attach aux loss
        # if torch.is_grad_enabled(): # only when grad enabled
        # with nvtx.annotate("attach_auxiliary_loss", color="blue"):
        if routed_expert_router_aux_loss_info is not None:
            # NOTE: this part cpu runtime > gpu runtime, so it's moved from directly after router_forward to here
            # because we need to avoid stalling the gpu stream
            # gpu stream is generally more ahead of cpu thread at the end of the block, hence less harmful to put it here
            routed_expert_router_aux_loss = self.routed_experts_router.compute_aux_loss(*routed_expert_router_aux_loss_info)

            # NOTE: the attach only writes 1.0 to the aux loss grad slot, so it should not matter where to attach
            final_out = attach_auxiliary_loss(final_out, routed_expert_router_aux_loss)

        return final_out

    def merge_shared_and_routed_out(
        self,
        shared_out: torch.Tensor,
        shared_factor: float,
        routed_out: torch.Tensor,
        routed_factor: float,
    ) -> torch.Tensor:
        # Combine shared and routed outputs
        return shared_out * shared_factor + routed_out * routed_factor

    def combined_forward_ep_no_sync(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with EP no-sync using symmetric-memory all_to_all_vdev(_2d) ops."""
        assert self.routed_experts is not None
        assert self.routed_experts_router is not None
        assert self.ep_enabled
        assert self.num_local_routed_experts is not None
        assert use_torch_grouped_mm() == True, "EP no-sync implementation requires torch.grouped_mm support"
        assert not requires_host_side_split_sizes(), "EP no-sync implementation does not support host-side split size communication"
        
        group_name = self._get_ep_no_sync_group_name()
        B, S, D = x.shape

        block_inp = x
        del x

        attn_res_out = self._checkpointed_res_norm_attn(block_inp, **kwargs)

        kwargs.pop("max_doc_len", None)
        kwargs.pop("cu_doc_lens", None)

        (
            local_x_global_routed_expert_weights,  # (B, S, top_k)
            local_x_global_routed_expert_indices,  # (B, S, top_k)
            local_batch_size_per_global_routed_expert,  # (num_experts, )
            routed_expert_router_aux_loss_info,
        ) = self.router_forward(
            router=self.routed_experts_router,
            local_x=attn_res_out,
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
                    local_x_global_shared_expert_weights,  # (B, S, E_shared)
                    _,
                    _,
                    _,
                ) = self.router_forward(
                    router=self.shared_experts_router,
                    local_x=attn_res_out,
                    scores_only=True,
                    loss_div_factor=loss_div_factor,
                )
            else:
                local_x_global_shared_expert_weights = None

        moe_inp = attn_res_out
        in_shape = moe_inp.size()
        moe_inp = moe_inp.view(-1, in_shape[-1])  # (B*S, D)

        num_out_tokens = local_x_global_routed_expert_indices.numel() # a constant

        with torch.no_grad():
            with nvtx.annotate("ConfigCapacity", color="green"):
                requested_splits = local_batch_size_per_global_routed_expert.to(dtype=torch.long)
                rank_capacity = self._compute_ep_no_sync_rank_capacity(num_out_tokens)
                if self.ep_no_sync_use_2d_all_to_all:
                    allowed_splits = cast(
                        torch.Tensor,
                        self._sync_tail_drop_allowed_splits_single_a2a(
                            requested_splits,
                            rank_capacity=rank_capacity,
                        ),
                    )
                    recv_splits_by_src_local = None
                else:
                    allowed_splits, recv_splits_by_src_local = cast(
                        Tuple[torch.Tensor, torch.Tensor],
                        self._sync_tail_drop_allowed_splits_single_a2a(
                            requested_splits,
                            rank_capacity=rank_capacity,
                            return_recv_splits_by_src_local=True,
                        ),
                    )

                local_reorder_indices, local_inverse_reorder_indices, packed_keep_mask = self._build_keep_reorder(
                    requested_splits=requested_splits,
                    keep_splits=allowed_splits,
                    num_out_tokens=num_out_tokens,
                )
                if self.ep_no_sync_use_2d_all_to_all:
                    align_pad = max(self.ep_no_sync_major_align - 1, 0)
                    dispatch_out_pad = (self.num_local_routed_experts - 1) * align_pad
                    combine_out_pad = (self.routed_experts_router.num_experts - 1) * align_pad
                else:
                    dispatch_out_pad = 0
                    combine_out_pad = 0

                dispatch_in_cap = num_out_tokens
                dispatch_out_cap = rank_capacity + dispatch_out_pad
                combine_out_cap = num_out_tokens + combine_out_pad

        with nvtx.annotate("Permute local tokens", color="green"):
            routing_map = local_x_global_routed_expert_indices.view(
                -1, self.routed_experts_router.top_k
            ).int()
            hidden_shape_before_permute = moe_inp.shape
            permutated_local_x, reversed_local_x_permutation_mapping = moe_permute_no_compile(
                inp=moe_inp,
                routing_map=routing_map,
                num_out_tokens=num_out_tokens,
                map_type="index",
            )

        buffers = self._get_ep_no_sync_buffers(
            dispatch_in_cap=dispatch_in_cap,
            dispatch_out_cap=dispatch_out_cap,
            combine_out_cap=combine_out_cap,
            d_model=permutated_local_x.shape[-1],
            dtype=permutated_local_x.dtype,
            device=permutated_local_x.device,
        )

        # lauch shared experts right before dispatch to overlap with the all_to_all
        if self.shared_experts is not None:
            wait_stream_no_compile(
                this_stream=self.get_dense_stream(),
                other_stream=torch.cuda.current_stream(),
            ) # type: ignore
            with torch.cuda.stream(self.get_dense_stream()):
                shared_out_up, shared_out_gate = self.shared_experts.forward1(attn_res_out)
        else:
            shared_out_up, shared_out_gate = None, None

        if self.ep_no_sync_use_2d_all_to_all:
            dispatch_out, dispatch_splits_offsets = _DispatchVDev2DAutograd.apply(
                permutated_local_x,
                local_reorder_indices,
                allowed_splits,
                buffers.dispatch_in,
                buffers.dispatch_in_splits,
                buffers.dispatch_out,
                buffers.dispatch_splits_offsets,
                buffers.dispatch_tmp_splits_offsets,
                group_name,
                self.ep_pg,
                self.ep_no_sync_major_align,
            )

            dispatch_splits = dispatch_splits_offsets[0]
            dispatch_offsets = dispatch_splits_offsets[1]
            
            with torch.no_grad():
                padded_batch_size_per_local_expert = self._build_padded_local_expert_batch_sizes_from_layout(
                    splits=dispatch_splits,
                    offsets=dispatch_offsets,
                    total_rows=dispatch_out.shape[0],
                )

            global_x_chunked = self.routed_experts(dispatch_out, padded_batch_size_per_local_expert)

            # NOTE: global_x_chunked may contain NaNs due to torch grouped_mm's behavior for pad postions (uninit values for > offsets[-1])
            buffers.dispatch_tmp_splits_offsets.copy_(dispatch_splits_offsets)

            # set shared experts to wait until this point to maximize overlap with alltoall
            wait_stream_no_compile(
                this_stream=self.get_dense_stream(),
                other_stream=torch.cuda.current_stream(),
            ) # type: ignore

            combine_out, combine_splits_offsets = _CombineVDev2DOffsetAutograd.apply(
                global_x_chunked,
                buffers.dispatch_tmp_splits_offsets,
                buffers.combine_in,
                buffers.combine_out,
                buffers.combine_splits_offsets,
                buffers.combine_tmp_splits_offsets,
                buffers.dispatch_in_splits,
                group_name,
                self.ep_pg,
                self.ep_no_sync_major_align,
            )

        else:
            assert recv_splits_by_src_local is not None
            with torch.no_grad():
                send_rank_splits = allowed_splits.view(
                    self.ep_world_size, self.num_local_routed_experts
                ).sum(dim=-1, dtype=torch.long)

            dispatch_out, dispatch_rank_splits_offsets = _DispatchVDevAutograd.apply(
                permutated_local_x,
                local_reorder_indices,
                send_rank_splits,
                buffers.dispatch_in,
                buffers.dispatch_in_rank_splits,
                buffers.dispatch_out,
                buffers.dispatch_rank_splits_offsets,
                buffers.dispatch_tmp_rank_splits_offsets,
                group_name,
                self.ep_pg,
            )

            dispatch_rank_major = dispatch_out # 1D Dispatch does not have holes, so no need to defrag

            with torch.no_grad():
                recv_splits_src_local_flat = recv_splits_by_src_local.reshape(-1)
                recv_chunk_ids, _, recv_valid_mask = _build_chunk_row_plan(
                    recv_splits_src_local_flat,
                    rows=dispatch_rank_major.shape[0],
                    device=dispatch_rank_major.device,
                )
                global_x_local_expert_indices = torch.remainder(
                    recv_chunk_ids, self.num_local_routed_experts
                ).to(dtype=torch.int32)
                # Tail rows beyond valid received tokens should not map to expert 0.
                # Route them to the last local expert so they remain after all valid expert segments.
                global_x_local_expert_indices = torch.where(
                    recv_valid_mask,
                    global_x_local_expert_indices,
                    torch.full_like(
                        global_x_local_expert_indices,
                        self.num_local_routed_experts - 1,
                    ),
                )

                # Per-local-expert rows received on this rank.
                padded_batch_size_per_local_expert = recv_splits_by_src_local.sum(
                    dim=0,
                    dtype=torch.long,
                )


            with nvtx.annotate("Permute global tokens", color='green'):
                routing_map2 = global_x_local_expert_indices.view(-1, 1).int()
                num_out_tokens2 = routing_map2.size(0)
                if self.routed_experts.num_local_experts == 1:
                    reversed_global_x_permutation_mapping = None
                else:
                    # moe_permute (by TE)
                    dispatch_rank_major, reversed_global_x_permutation_mapping = moe_permute_no_compile(
                        inp=dispatch_rank_major, 
                        routing_map=routing_map2, 
                        num_out_tokens=num_out_tokens2, 
                        map_type='index'
                    )

            # expert forward
            dispatch_rank_major = self.routed_experts(dispatch_rank_major, padded_batch_size_per_local_expert)
            

            with nvtx.annotate("Unpermute global tokens", color='green'):
                if self.routed_experts.num_local_experts == 1:
                    global_x_rank_major = dispatch_rank_major  # skip unpermute if only one local expert
                else:
                    # moe_unpermute (by TE)
                    global_x_rank_major = moe_unpermute_no_compile(
                        inp=dispatch_rank_major,
                        row_id_map=reversed_global_x_permutation_mapping,
                        merging_probs=None,
                        restore_shape=None, # map_type='index' does not require restore_shape
                        map_type='index',
                    ) 


            # set shared experts to wait until this point to maximize overlap with alltoall
            wait_stream_no_compile(
                this_stream=self.get_dense_stream(),
                other_stream=torch.cuda.current_stream(),
            ) # type: ignore

            combine_out, combine_rank_splits_offsets = _CombineVDevAutograd.apply(
                global_x_rank_major,
                dispatch_rank_splits_offsets[0],
                buffers.combine_in,
                buffers.combine_in_rank_splits,
                buffers.combine_out,
                buffers.combine_rank_splits_offsets,
                buffers.combine_tmp_rank_splits_offsets,
                group_name,
                self.ep_pg,
            ) 


        # launch second half of the shared expert forward
        if self.shared_experts is not None:
            assert shared_out_up is not None
            assert shared_out_gate is not None

            with torch.cuda.stream(self.get_dense_stream()):
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

        if self.ep_no_sync_use_2d_all_to_all:
            # 2D restore = defrag + inverse reorder + zero out dropped tokens
            restored_local_x = self._restore_local_from_chunked_combine_out(
                combine_out,
                combine_splits_offsets=combine_splits_offsets,
                local_inverse_reorder_indices=local_inverse_reorder_indices,
                packed_keep_mask=packed_keep_mask,
                out_rows=num_out_tokens,
            )
        else:
            # 1D Combine does not have holes, so no need to defrag
            # inverse reorder and zero out dropped tokens
            with nvtx.annotate("RestoreDrop", color="green"):
                restored_local_x = combine_out.index_select(0, local_inverse_reorder_indices)
                restored_keep_mask = packed_keep_mask.index_select(0, local_inverse_reorder_indices)
                restored_local_x = torch.where(
                    restored_keep_mask.unsqueeze(-1),
                    restored_local_x,
                    torch.zeros_like(restored_local_x),
                )

        with nvtx.annotate("Unpermute-Merge local tokens", color="green"):
            local_x = moe_unpermute_no_compile(
                inp=restored_local_x,
                row_id_map=reversed_local_x_permutation_mapping,
                merging_probs=local_x_global_routed_expert_weights.view(-1, self.routed_experts_router.top_k),
                restore_shape=hidden_shape_before_permute,
                map_type="index",
            )
            local_x = cast(torch.Tensor, local_x)

        local_x = local_x.view(in_shape)
        wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

        if self.shared_experts is not None:
            assert mixed_shared_out is not None
            mlp_out = local_x + mixed_shared_out
        else:
            mlp_out = local_x

        final_out = attn_res_out + self.feed_forward_norm(mlp_out)

        if routed_expert_router_aux_loss_info is not None:
            routed_expert_router_aux_loss = self.routed_experts_router.compute_aux_loss(
                *routed_expert_router_aux_loss_info
            )
            final_out = attach_auxiliary_loss(final_out, routed_expert_router_aux_loss)


        return final_out

    def combined_forward_ep(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function with EP"""
        assert self.routed_experts_router is not None
        assert self.ep_enabled == True
        assert self.num_local_routed_experts is not None


        B, S, D = x.shape

        # rename "x" to avoid confusion
        block_inp = x
        del x

        # attention 
        # + attention norm
        # + residual connection
        # attn_res_out = block_inp + self.attention_norm(self.attention(block_inp, **kwargs))
        attn_res_out = self._checkpointed_res_norm_attn(block_inp, **kwargs)

        # remove attention kwargs
        kwargs.pop("max_doc_len", None)
        kwargs.pop("cu_doc_lens", None)


        # routed expert router
        (
            local_x_global_routed_expert_weights, # (B, S, top_k)
            local_x_global_routed_expert_indices, # (B, S, top_k)
            local_batch_size_per_global_routed_expert, # (num_experts, )
            routed_expert_router_aux_loss_info # tuple
        ) = self.router_forward(
            router=self.routed_experts_router,
            local_x=attn_res_out, 
            scores_only=False,
            loss_div_factor=loss_div_factor # scalar
        )
        

        # the shared experts (executed on the dense stream) need to wait for `attn_res_out` and `local_x_global_shared_expert_weights` (on the main stream) to be complete
        wait_stream_no_compile(
            this_stream=self.get_dense_stream(),  
            other_stream=torch.cuda.current_stream() 
        ) 

        #### Communicate the number of tokens that will be sent to each device ###
        with nvtx.annotate("Token count all_to_all", color='green'):
            with torch.no_grad():
                # Pass token count information to the device on which the
                # target expert resides.
                global_batch_size_per_local_expert = torch.empty_like(
                    local_batch_size_per_global_routed_expert,
                )
                global_batch_size_handle = dist.all_to_all_single(
                    global_batch_size_per_local_expert, # Gathered concatenated output tensor.
                    local_batch_size_per_global_routed_expert, # Input tensor to scatter.
                    group=self.ep_pg,
                    async_op=True,
                )
                # NOTE:
                # local_batch_size_per_global_routed_expert: 
                # (num_experts, ) -> view as: (EP, num_local_routed_experts)
                #   data[i] = how many tokens should go to global expert i (can be on other rank)
                # global_batch_size_per_local_expert: 
                # (num_experts, ) -> view as: (EP, num_local_routed_experts)
                #   data[i][j] = how many tokens from rank i will go to local expert j on this rank
                assert global_batch_size_handle is not None # because of async

        ### Optionally run shared experts router ###
        with torch.cuda.stream(self.get_dense_stream()):
            if self.shared_experts_router:
                # shared expert router
                (
                    local_x_global_shared_expert_weights, # (B, S, E_shared)
                    _, 
                    _, 
                    _ 
                ) = self.router_forward(
                    router=self.shared_experts_router,
                    local_x=attn_res_out, 
                    scores_only=True,  # only need scores for shared experts
                    loss_div_factor=loss_div_factor # scalar
                )
            else:
                local_x_global_shared_expert_weights = None
        
        
        moe_inp = attn_res_out

        in_shape = moe_inp.size()
        
        moe_inp = moe_inp.view(-1, in_shape[-1])  # (B*S, D)


        ### Configure the sizes for grouped GEMM ###

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with nvtx.annotate("Sync token count", color='green'):
            with torch.no_grad():
                global_batch_size_handle.wait()

                # Reshape to (ep_world_size, num_local_routed_experts).
                local_batch_size_per_global_routed_expert = local_batch_size_per_global_routed_expert.view(
                    self.ep_world_size, self.num_local_routed_experts
                )
                global_batch_size_per_local_expert = global_batch_size_per_local_expert.view(
                    self.ep_world_size, self.num_local_routed_experts
                )
                # Calculate the bins boundaries from the token counts. # [EP, num_local_routed_experts] -> [num_local_routed_experts,]
                parallel_batch_size_per_local_expert = global_batch_size_per_local_expert.sum(
                    dim=0,
                    dtype=torch.long,
                )
                
                send_counts_gpu = local_batch_size_per_global_routed_expert.sum(dim=-1)
                recv_counts_gpu = global_batch_size_per_local_expert.sum(dim=-1)
                # NOTE: host-device sync here.
                send_counts_cpu, copy_stream, dtoh_event_send = async_copy_to_cpu(send_counts_gpu, event=self._dtoh_event_send)  
                recv_counts_cpu, copy_stream, dtoh_event_recv = async_copy_to_cpu(recv_counts_gpu, event=self._dtoh_event_recv) 
                parallel_batch_size_per_local_expert_cpu, copy_stream, dtoh_event = async_copy_to_cpu(parallel_batch_size_per_local_expert, event=self._dtoh_event)  


        with torch.no_grad():
            # Construct the expert indices for the permuted tokens.
            global_x_local_expert_indices = torch.remainder(
                torch.arange(
                    self.routed_experts_router.num_experts,
                    dtype=torch.int32,
                    device=moe_inp.device,
                ),
                self.num_local_routed_experts,
            ) # e.g. [0, 1, 2, 3, 0, 1, 2, 3, ...] for 4 local experts

        ### permute local tokens to be ready for all-to-all communication ###
        with nvtx.annotate("Permute local tokens", color='green'):
            routing_map = local_x_global_routed_expert_indices.view(-1, self.routed_experts_router.top_k).int()
            num_out_tokens = routing_map.size(0) * self.routed_experts_router.top_k # dropless
            hidden_shape_before_permute = moe_inp.shape
            permutated_local_x, reversed_local_x_permutation_mapping = moe_permute_no_compile(
                inp=moe_inp, 
                routing_map=routing_map, 
                num_out_tokens=num_out_tokens, 
                map_type='index'
            ) 
            
            # now permutated_local_x tokens are grouped by expert, which means tokens will go to expert id:
            # [0 , 0 , ... 1, ... 2, ..., ..., 31, 31]  (if 32 experts)
            # if EP=8, each rank has 4 experts, then tokens of
            # [0, 0, ..., 3, 3] go to rank 0,
            # [4, 4, ..., 7, 7] go to rank 1, 
            # and so on.

        ### Optionally run shared experts to overlap with the all-to-all communication ###
        if self.shared_experts is not None:
            # overlap compute while waiting for all2all
            wait_stream_no_compile(
                this_stream=self.get_dense_stream(), 
                other_stream=torch.cuda.current_stream()
            )
            with torch.cuda.stream(self.get_dense_stream()):
                shared_out_up, shared_out_gate = self.shared_experts.forward1(attn_res_out)
                # shared_out = self.shared_experts.forward(attn_res_out)
                # NOTE: the shared_experts forward is queued, but will not start to run until the DtoH is done
        else:
             shared_out_up, shared_out_gate = None, None

        ### wait for the DtoH to complete ###
        with torch.no_grad():
            assert dtoh_event_send
            assert dtoh_event_recv
            assert dtoh_event

            dtoh_event.synchronize() # ensure `parallel_batch_size_per_local_expert_cpu` is ready
            send_counts = send_counts_cpu.tolist() # tensor to list
            recv_counts = recv_counts_cpu.tolist() # tensor to list
            # if 0 in send_counts:
            #     print(f"[Warning] block {self.block_idx} EP rank {get_rank(self.ep_pg)} has 0 send counts: {send_counts}")
            # if 0 in recv_counts:
            #     print(f"[Warning] block {self.block_idx} EP rank {get_rank(self.ep_pg)} has 0 recv counts: {recv_counts}")
            tokens_received = sum(recv_counts)

        if tokens_received == 0: # make sure it's not wrapped in torch.no_grad() so that it can be printed even when grad enabled (used for debugging)
            print(f"[Warning] (grad={torch.is_grad_enabled()}) block {self.block_idx} EP rank {get_rank(self.ep_pg)} has 0 tokens received in all2all: send_counts={send_counts} recv_counts={recv_counts}")

        ### Start the all-to-all communication asynchronously ###
        with nvtx.annotate("all2all", color='green'):
            permutated_local_x, global_x, global_x_handle = ops.all_to_all_async(
                permutated_local_x,
                recv_counts,
                send_counts,
                group=self.ep_pg,
            )

        with torch.no_grad():
            # this specifiyes for the received global tokens, which local expert they belong to
            global_x_local_expert_indices = torch.repeat_interleave(
                global_x_local_expert_indices,
                global_batch_size_per_local_expert.flatten(),
                output_size=tokens_received,
            ) # e.g. [0, ...,  0, ... , 3, ..., 3, 0, ...] for 4 local experts
        

        global_x = ops.all_to_all_wait(permutated_local_x, global_x, global_x_handle)

        ### global_permute + routed experts forward + glboal unpermute ###
        global_x = self._checkpointed_permute_routed_experts_unpermute(
            global_x=global_x,
            global_x_local_expert_indices=global_x_local_expert_indices,
            parallel_batch_size_per_local_expert_cpu=parallel_batch_size_per_local_expert_cpu if requires_host_side_split_sizes() else parallel_batch_size_per_local_expert,
        )

        # reverse_all_to_all 
        before_rev_all2all_event = torch.cuda.current_stream().record_event(
            event=self._before_rev_all2all_event # type: ignore
        ) 
        with nvtx.annotate("reverse_all_to_all", color='green'):
            global_x = cast(torch.Tensor, global_x)

            global_x, local_x, local_x_handle = ops.all_to_all_async(
                global_x,
                send_counts,
                recv_counts,
                group=self.ep_pg,
            )

        if self.shared_experts is not None:
            # variables from forward1
            assert shared_out_up is not None
            assert shared_out_gate is not None

            # the `merge_shared` should not start until the start of the reverse all2all to better overlap it
            # before_rev_all2all_event.wait(self.get_dense_stream())  # NOTE: this raises "torch.AcceleratorError: CUDA error: invalid device ordinal" error. Likely a torch bug of Dynamo in 2.10.0; error only in Dynamo, not in eager.
            self.get_dense_stream().wait_event(before_rev_all2all_event) # this does not error. Weird.
            # merge shared experts when waiting for all2all
            with nvtx.annotate("merge_shared", color='purple'):
                with torch.cuda.stream(self.get_dense_stream()):

                    shared_out = self.shared_experts.forward2(shared_out_up, shared_out_gate, attn_res_out.shape)
                    if self.shared_experts_router:
                        assert local_x_global_shared_expert_weights is not None
                        # weighted sum of the shared experts by router weights
                        # local_x_global_shared_expert_weights -> (B, S, E_shared)
                        # shared_out -> (E_shared, B, S, D)
                        _, _, E_s = local_x_global_shared_expert_weights.shape
                        local_x_global_shared_expert_weights.shape
                        mixed_shared_out = torch.bmm(
                            local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(B*S, 1, E_s),            # (BS, 1, E), 
                            shared_out.permute(1, 2, 0, 3).contiguous().view(B*S, E_s, D)              # (BS, E, D)
                        ).squeeze(1).view(B, S, D)
                    else:
                        mixed_shared_out = shared_out.squeeze(0)
        else:
            mixed_shared_out = None


        local_x = ops.all_to_all_wait(global_x, local_x, local_x_handle)

        #### Unpermute the (local) tokens returned by all-to-all communication 
        with nvtx.annotate("Unpermute-Merge local tokens", color='green'):
            if self.checkpoint_second_unpermute:
                local_x = checkpoint(
                    moe_unpermute_no_compile,
                    inp=local_x,
                    row_id_map=reversed_local_x_permutation_mapping,
                    merging_probs=local_x_global_routed_expert_weights.view(-1, self.routed_experts_router.top_k),
                    restore_shape=hidden_shape_before_permute,
                    map_type='index',
                    use_reentrant=False
                )
                local_x = cast(torch.Tensor, local_x)
            else:
                local_x = moe_unpermute_no_compile(
                    inp=local_x,
                    row_id_map=reversed_local_x_permutation_mapping,
                    merging_probs=local_x_global_routed_expert_weights.view(-1, self.routed_experts_router.top_k),
                    restore_shape=hidden_shape_before_permute,
                    map_type='index',
                ) # type: ignore
                local_x = cast(torch.Tensor, local_x)

        ####
    
        
        local_x = local_x.view(in_shape)

        # need to use `mixed_shared_out`
        wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream()) 


        # weighted sum of the shared experts and routed experts
        if self.shared_experts is not None:
            assert mixed_shared_out is not None
            mlp_out = local_x + mixed_shared_out
        else:
            mlp_out = local_x

        final_out = attn_res_out + self.feed_forward_norm(mlp_out)

        ####

        # attach aux loss
        # if torch.is_grad_enabled(): # only when grad enabled
        # with nvtx.annotate("attach_auxiliary_loss", color="blue"):
        if routed_expert_router_aux_loss_info is not None:
            # NOTE: this part cpu runtime > gpu runtime, so it's moved from directly after router_forward to here
            # because we need to avoid stalling the gpu stream
            # gpu stream is generally more ahead of cpu thread at the end of the block, hence less harmful to put it here
            routed_expert_router_aux_loss = self.routed_experts_router.compute_aux_loss(*routed_expert_router_aux_loss_info)

            # NOTE: the attach only writes 1.0 to the aux loss grad slot, so it should not matter where to attach
            final_out = attach_auxiliary_loss(final_out, routed_expert_router_aux_loss)
                    
        # torch.save( {
        #     'final_out': final_out.detach().cpu(),
        # },
        # f'ep_sync_debug_block_{self.block_idx}_rank_{get_rank(self.ep_pg)}.pt')
        return final_out
    


    def _routed_experts_unpermute(
        self,
        global_x,
        global_x_local_expert_indices,
        parallel_batch_size_per_local_expert_cpu,
        hidden_shape_before_permute2,
        reversed_global_x_permutation_mapping,
    ):
        assert self.routed_experts is not None

        ## MLP forwrad ##
        global_x = self.routed_experts(global_x, parallel_batch_size_per_local_expert_cpu)
        
        ## Unpermute the output tokens to be ready for all-to-all communication ##
        with nvtx.annotate("Unpermute global tokens", color='green'):
            if self.routed_experts.num_local_experts == 1:
                global_x_restore = global_x  # skip unpermute if only one local expert
            else:
                # option 1: use moe_sort_chunks_by_index (by TE <- trition)
                # deprecated (code removed)

                # option 2: use moe_unpermute (by TE)
                global_x_restore = moe_unpermute_no_compile(
                    inp=global_x,
                    row_id_map=reversed_global_x_permutation_mapping,
                    merging_probs=None,
                    restore_shape=hidden_shape_before_permute2,
                    map_type='index',
                ) 


        return global_x_restore


    def _checkpointed_permute_routed_experts_unpermute(
        self,
        global_x,
        global_x_local_expert_indices,
        parallel_batch_size_per_local_expert_cpu
    ) -> torch.Tensor:
        # don't need to checkpoint the permute step because it does not save input for backward

        ##  5. Permute the global tokens to be ready for MLP computation ##
        with nvtx.annotate("Permute global tokens for MLP", color='green'):
            # option 1: use moe_sort_chunks_by_index (by TE <- trition)
            # deprecated (code removed)

            # option 2: use moe_permute (by TE), and pretend topk is 1
            routing_map2 = global_x_local_expert_indices.view(-1, 1).int()
            num_out_tokens2 = routing_map2.size(0) * 1 # dropless
            hidden_shape_before_permute2 = global_x.shape
            if self.routed_experts.num_local_experts == 1:
                reversed_global_x_permutation_mapping = None
            else:
                global_x, reversed_global_x_permutation_mapping = moe_permute_no_compile(
                    inp=global_x, 
                    routing_map=routing_map2, 
                    num_out_tokens=num_out_tokens2, 
                    map_type='index'
                )

        if self.checkpoint_permute_moe_unpermute:
            out = checkpoint(
                self._routed_experts_unpermute, 
                global_x,
                global_x_local_expert_indices,
                parallel_batch_size_per_local_expert_cpu, 
                hidden_shape_before_permute2,
                reversed_global_x_permutation_mapping,
                use_reentrant=False, 
            )
            return cast(torch.Tensor, out)
        else:
            return self._routed_experts_unpermute(
                global_x,
                global_x_local_expert_indices,
                parallel_batch_size_per_local_expert_cpu, 
                hidden_shape_before_permute2,
                reversed_global_x_permutation_mapping
            )

    
    def _res_norm_attn(
        self,
        block_inp,
        **kwargs
    ) -> torch.Tensor:
        attn_res_out = block_inp + self.attention_norm(self.attention(block_inp, **kwargs))
        return attn_res_out

    def _checkpointed_res_norm_attn(
        self,
        block_inp,
        **kwargs
    ) -> torch.Tensor:
        if self.checkpoint_attn:
            out = checkpoint(
                self._res_norm_attn,
                block_inp,
                use_reentrant=False,
                **kwargs,
            )
            return cast(torch.Tensor, out)
        else:
            return self._res_norm_attn(block_inp, **kwargs)


    def post_batch(self, dry_run: bool = False):
        """
        Should be called right after the final backward of a complete batch but before the optimizer step.
        """
        if self.shared_experts_router:
            self.shared_experts_router.post_batch(dry_run=dry_run)
        if self.routed_experts_router:
            self.routed_experts_router.post_batch(dry_run=dry_run)


class MoEFusedV2TransformerTBOBlock(MoEFusedV2TransformerBlock):

    def combined_forward_ep_tbo(
        self,
        x0: torch.Tensor,
        x1_ctx: Dict,
        x1_is_fresh: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        assert self.routed_experts is not None
        assert self.routed_experts_router is not None
        assert self.ep_enabled == True
        assert self.num_local_routed_experts is not None

        B, S, D = x0.shape


        # rename "x" to avoid confusion
        block_inp = x0
        del x0

        with torch.no_grad():
            # Construct the expert indices for the permuted tokens.
            global_x_local_expert_indices_0 = torch.remainder(
                torch.arange(
                    self.routed_experts.num_experts,
                    dtype=torch.int32,
                    device=block_inp.device,
                ),
                self.num_local_routed_experts,
            ) # e.g. [0, 1, 2, 3, 0, 1, 2, 3, ...] for 4 local experts
        

        ############################ TBO 1 ############################
        with nvtx.annotate("TBO-1", color='orange'):
            if x1_is_fresh:
                local_x1, local_x_handle1 = None, None
                last_block = None
            else:
                global_x1 = x1_ctx['global_x1']
                send_counts1 = x1_ctx['send_counts1']
                recv_counts1 = x1_ctx['recv_counts1']
                # tokens_received1 = x1_ctx['tokens_received1']

                last_block = cast(MoEFusedV2TransformerBlock, x1_ctx['last_block'])

                assert last_block.routed_experts_router is not None
                # finish reverse all2all and other ops for x1
                with nvtx.annotate("reverse_all_to_all", color='green'):
                    global_x1 = cast(torch.Tensor, global_x1)
                    global_x1, local_x1, local_x_handle1 = ops.all_to_all_async(
                    # local_x1, local_x_handle1 = ops.all_to_all(
                        global_x1,
                        send_counts1,
                        recv_counts1,
                        group=last_block.ep_pg,
                        # async_op=True,
                    )


        ############################ END: TBO 1 ########################

        with nvtx.annotate("TBO-0", color='purple'):
            # attention 
            # + attention norm
            # + residual connection
            # attn_res_out = block_inp + self.attention_norm(self.attention(block_inp, **kwargs))
            attn_res_out = self._checkpointed_res_norm_attn(block_inp, **kwargs)
            # routed expert router
            (
                local_x_global_routed_expert_weights, # (B, S, top_k)
                local_x_global_routed_expert_indices, # (B, S, top_k)
                local_batch_size_per_global_routed_expert, # (num_experts, )
                routed_expert_router_aux_loss # scalar # TODO: update code
            ) = self.router_forward(
                router=self.routed_experts_router,
                local_x=attn_res_out, 
                scores_only=False,
                loss_div_factor=loss_div_factor # scalar
            )
        
            # attach aux loss
            if torch.is_grad_enabled(): # only when grad enabled
                with nvtx.annotate("attach_auxiliary_loss", color="blue"):
                    if routed_expert_router_aux_loss is not None:
                        attn_res_out = attach_auxiliary_loss(attn_res_out, routed_expert_router_aux_loss) # TODO: update code


            ########### 1. Communicate the number of tokens that will be sent to each device ###########
            with nvtx.annotate("Token count all_to_all", color='green'):
                with torch.no_grad():
                    # Pass token count information to the device on which the
                    # target expert resides.
                    global_batch_size_per_local_expert = torch.empty_like(
                        local_batch_size_per_global_routed_expert,
                    )
                    global_batch_size_handle = dist.all_to_all_single(
                        global_batch_size_per_local_expert, # Gathered concatenated output tensor.
                        local_batch_size_per_global_routed_expert, # Input tensor to scatter.
                        group=self.ep_pg,
                        async_op=True,
                    )

                    assert global_batch_size_handle is not None # because of async

            ############################################ end

            if self.shared_experts_router:
                # shared expert router
                (
                    local_x_global_shared_expert_weights, # (B, S, E_shared)
                    _, 
                    _, 
                    _ 
                ) = self.router_forward(
                    router=self.shared_experts_router,
                    local_x=attn_res_out, 
                    scores_only=True,  # only need scores for shared experts
                    loss_div_factor=loss_div_factor # scalar
                )
            else:
                local_x_global_shared_expert_weights = None
            

            # forward shared experts
            if self.shared_experts is not None:
                shared_out = self.shared_experts.forward(attn_res_out)
                with nvtx.annotate("merge_shared", color='purple'):
                    # shared_out = self.shared_experts.forward2(shared_out_up, shared_out_gate, attn_res_out.shape)
                    if self.shared_experts_router:
                        assert local_x_global_shared_expert_weights is not None
                        # weighted sum of the shared experts by router weights
                        # local_x_global_shared_expert_weights -> (B, S, E_shared)
                        # shared_out -> (E_shared, B, S, D)
                        _, _, E_s = local_x_global_shared_expert_weights.shape
                        local_x_global_shared_expert_weights.shape
                        mixed_shared_out = torch.bmm(
                            local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(B*S, 1, E_s),            # (BS, 1, E), 
                            shared_out.permute(1, 2, 0, 3).contiguous().view(B*S, E_s, D)              # (BS, E, D)
                        ).squeeze(1).view(B, S, D)
                    else:
                        mixed_shared_out = shared_out.squeeze(0)
            else:
                mixed_shared_out = None

            
            moe_inp = attn_res_out

            in_shape = moe_inp.size()
            
            moe_inp = moe_inp.view(-1, in_shape[-1])  # (B*S, D)


            ###########  3. Configure the sizes for grouped GEMM ###########

            # Compute the number of tokens that will be received from each
            # device and permute the input data across the devices.
            with nvtx.annotate("Sync token count", color='green'):
                with torch.no_grad():
                    global_batch_size_handle.wait()

                    # Reshape to (ep_world_size, num_local_routed_experts).
                    local_batch_size_per_global_routed_expert = local_batch_size_per_global_routed_expert.view(
                        self.ep_world_size, self.num_local_routed_experts
                    )
                    global_batch_size_per_local_expert = global_batch_size_per_local_expert.view(
                        self.ep_world_size, self.num_local_routed_experts
                    )
                    # Calculate the bins boundaries from the token counts. # [EP, num_local_routed_experts] -> [num_local_routed_experts,]
                    parallel_batch_size_per_local_expert = global_batch_size_per_local_expert.sum(
                        dim=0,
                        dtype=torch.long,
                    )
                    

                    send_counts_gpu = local_batch_size_per_global_routed_expert.sum(dim=-1)
                    recv_counts_gpu = global_batch_size_per_local_expert.sum(dim=-1)
                    send_counts_cpu, _, dtoh_event_send = async_copy_to_cpu(send_counts_gpu, event=self._dtoh_event_send)  
                    recv_counts_cpu, _, dtoh_event_recv = async_copy_to_cpu(recv_counts_gpu, event=self._dtoh_event_recv) 
                    parallel_batch_size_per_local_expert_cpu, _, dtoh_event = async_copy_to_cpu(parallel_batch_size_per_local_expert, event=self._dtoh_event)  



            ########### 2. permute local tokens to be ready for all-to-all communication ###########

            with nvtx.annotate("Permute local tokens", color='green'):
                routing_map = local_x_global_routed_expert_indices.view(-1, self.routed_experts_router.top_k).int()
                num_out_tokens = routing_map.size(0) * self.routed_experts_router.top_k # dropless
                hidden_shape_before_permute = moe_inp.shape
                permutated_local_x, reversed_local_x_permutation_mapping = moe_permute_no_compile(
                    inp=moe_inp, 
                    routing_map=routing_map, 
                    num_out_tokens=num_out_tokens, 
                    map_type='index'
                ) 


            with torch.no_grad():
                # torch.cuda.current_stream().synchronize() # wait for the copy to CPU to finish
                assert dtoh_event_send
                assert dtoh_event_recv
                assert dtoh_event
                # dtoh_event_send.synchronize()
                # dtoh_event_recv.synchronize()
                dtoh_event.synchronize()
                send_counts = send_counts_cpu.tolist() # tensor to list
                recv_counts = recv_counts_cpu.tolist() # tensor to list
                tokens_received = sum(recv_counts)

            with nvtx.annotate("all2all", color='green'):
                permutated_local_x, global_x, global_x_handle = ops.all_to_all_async(
                    permutated_local_x,
                    recv_counts,
                    send_counts,
                    group=self.ep_pg,
                )

            with torch.no_grad():
                # this specifiyes for the received global tokens, which local expert they belong to
                global_x_local_expert_indices = torch.repeat_interleave(
                    global_x_local_expert_indices_0,
                    global_batch_size_per_local_expert.flatten(),
                    output_size=tokens_received,
                ) # e.g. [0, ...,  0, ... , 3, ..., 3, 0, ...] for 4 local experts

        with nvtx.annotate("TBO-1", color='orange'):
            if x1_is_fresh:
                x1 = x1_ctx['x1']
                assert x1.shape == (B, S, D)
                block_inp1 = x1
                del x1
            else:
                reversed_local_x_permutation_mapping1 = x1_ctx['reversed_local_x_permutation_mapping1']
                local_x_global_routed_expert_weights1 = x1_ctx['local_x_global_routed_expert_weights1']
                hidden_shape_before_permute1 = x1_ctx['hidden_shape_before_permute1']
                in_shape1 = x1_ctx['in_shape1']
                mixed_shared_out1 = x1_ctx['mixed_shared_out1']
                attn_res_out1 = x1_ctx['attn_res_out1']
                
                assert last_block is not None
                assert local_x_handle1 is not None
                assert local_x1 is not None
                assert last_block.routed_experts_router is not None
                
                # local_x_handle1.wait()
                local_x1 = ops.all_to_all_wait(global_x1, local_x1, local_x_handle1)

                ## 9. Unpermute the (local) tokens returned by all-to-all communication ##
                with nvtx.annotate("Unpermute-Merge local tokens", color='green'):
                    local_x1 = moe_unpermute_no_compile(
                        inp=local_x1,
                        row_id_map=reversed_local_x_permutation_mapping1,
                        merging_probs=local_x_global_routed_expert_weights1.view(-1, last_block.routed_experts_router.top_k),
                        restore_shape=hidden_shape_before_permute1,
                        map_type='index',
                    )
                ## end
            
                
                local_x1 = local_x1.view(in_shape1)

                # weighted sum of the shared experts and routed experts
                if last_block.shared_experts is not None:
                    assert mixed_shared_out1 is not None
                    assert last_block.routed_experts is not None
                    mlp_out1 = local_x1 + mixed_shared_out1
                else:
                    mlp_out1 = local_x1

                block_inp1 = attn_res_out1 + last_block.feed_forward_norm(mlp_out1)
            
            ########## x1 last step done ##########

            # attention 
            # + attention norm
            # + residual connection
            # attn_res_out1 = block_inp1 + self.attention_norm(self.attention(block_inp1, **kwargs))
            attn_res_out1 = self._checkpointed_res_norm_attn(block_inp1, **kwargs)

            # routed expert router
            (
                local_x_global_routed_expert_weights1, # (B, S, top_k)
                local_x_global_routed_expert_indices1, # (B, S, top_k)
                local_batch_size_per_global_routed_expert1, # (num_experts, )
                routed_expert_router_aux_loss1 # scalar # TODO: update code
            ) = self.router_forward(
                router=self.routed_experts_router,
                local_x=attn_res_out1, 
                scores_only=False,
                loss_div_factor=loss_div_factor # scalar
            )
            
            # attach aux loss
            if torch.is_grad_enabled(): # only when grad enabled
                with nvtx.annotate("attach_auxiliary_loss", color="blue"):
                    if routed_expert_router_aux_loss1 is not None: # TODO: update code
                        attn_res_out1 = attach_auxiliary_loss(attn_res_out1, routed_expert_router_aux_loss1)
            


            with nvtx.annotate("Token count all_to_all", color='green'):
                with torch.no_grad():
                    # Pass token count information to the device on which the
                    # target expert resides.
                    global_batch_size_per_local_expert1 = torch.empty_like(
                        local_batch_size_per_global_routed_expert1,
                    )
                    global_batch_size_handle1 = dist.all_to_all_single(
                        global_batch_size_per_local_expert1, # Gathered concatenated output tensor.
                        local_batch_size_per_global_routed_expert1, # Input tensor to scatter.
                        group=self.ep_pg,
                        async_op=True,
                    )

                    assert global_batch_size_handle1 is not None # because of async



            if self.shared_experts_router:
                # shared expert router
                (
                    local_x_global_shared_expert_weights1, # (B, S, E_shared)
                    _, 
                    _, 
                    _ 
                ) = self.router_forward(
                    router=self.shared_experts_router,
                    local_x=attn_res_out1, 
                    scores_only=True,  # only need scores for shared experts
                    loss_div_factor=loss_div_factor # scalar
                )
            else:
                local_x_global_shared_expert_weights1 = None
            

            if self.shared_experts is not None:
                shared_out1 = self.shared_experts.forward(attn_res_out1)
                
                with nvtx.annotate("merge_shared", color='purple'):
                    if self.shared_experts_router:
                        assert local_x_global_shared_expert_weights1 is not None
                        # weighted sum of the shared experts by router weights
                        # local_x_global_shared_expert_weights -> (B, S, E_shared)
                        # shared_out -> (E_shared, B, S, D)
                        _, _, E_s1 = local_x_global_shared_expert_weights1.shape
                        local_x_global_shared_expert_weights1.shape
                        mixed_shared_out1 = torch.bmm(
                            local_x_global_shared_expert_weights1.to(shared_out1.dtype).reshape(B*S, 1, E_s1),            # (BS, 1, E), 
                            shared_out1.permute(1, 2, 0, 3).contiguous().view(B*S, E_s1, D)              # (BS, E, D)
                        ).squeeze(1).view(B, S, D)
                    else:
                        mixed_shared_out1 = shared_out1.squeeze(0)
            else:
                mixed_shared_out1 = None
            
            moe_inp1 = attn_res_out1

            in_shape1 = moe_inp1.size()
            
            moe_inp1 = moe_inp1.view(-1, in_shape1[-1])  # (B*S, D)


            ###########  3. Configure the sizes for grouped GEMM ###########

            # Compute the number of tokens that will be received from each
            # device and permute the input data across the devices.
            with nvtx.annotate("Sync token count", color='green'):
                with torch.no_grad():
                    global_batch_size_handle1.wait()

                    # Reshape to (ep_world_size, num_local_routed_experts).
                    local_batch_size_per_global_routed_expert1 = local_batch_size_per_global_routed_expert1.view(
                        self.ep_world_size, self.num_local_routed_experts
                    )
                    global_batch_size_per_local_expert1 = global_batch_size_per_local_expert1.view(
                        self.ep_world_size, self.num_local_routed_experts
                    )
                    # Calculate the bins boundaries from the token counts. # [EP, num_local_routed_experts] -> [num_local_routed_experts,]
                    parallel_batch_size_per_local_expert1 = global_batch_size_per_local_expert1.sum(
                        dim=0,
                        dtype=torch.long,
                    )
                    

                    send_counts_gpu1 = local_batch_size_per_global_routed_expert1.sum(dim=-1)
                    recv_counts_gpu1 = global_batch_size_per_local_expert1.sum(dim=-1)
                    send_counts_cpu1, _, dtoh_event_send1 = async_copy_to_cpu(send_counts_gpu1, event=self._dtoh_event_send1)  
                    recv_counts_cpu1, _, dtoh_event_recv1 = async_copy_to_cpu(recv_counts_gpu1, event=self._dtoh_event_recv1) 
                    parallel_batch_size_per_local_expert_cpu1, _, dtoh_event1 = async_copy_to_cpu(parallel_batch_size_per_local_expert1, event=self._dtoh_event1)  


            ########### 2. permute local tokens to be ready for all-to-all communication ###########

            with nvtx.annotate("Permute local tokens", color='green'):
                routing_map1 = local_x_global_routed_expert_indices1.view(-1, self.routed_experts_router.top_k).int()
                num_out_tokens1 = routing_map1.size(0) * self.routed_experts_router.top_k # dropless
                hidden_shape_before_permute1 = moe_inp1.shape
                permutated_local_x1, reversed_local_x_permutation_mapping1 = moe_permute_no_compile(
                    inp=moe_inp1, 
                    routing_map=routing_map1, 
                    num_out_tokens=num_out_tokens1, 
                    map_type='index'
                ) 

            #### end



            with torch.no_grad():
                # torch.cuda.current_stream().synchronize() # wait for the copy to CPU to finish
                # assert dtoh_event_send1
                # assert dtoh_event_recv1
                assert dtoh_event1
                # dtoh_event_send1.synchronize()
                # dtoh_event_recv1.synchronize()
                dtoh_event1.synchronize()
                send_counts1 = send_counts_cpu1.tolist() # tensor to list
                recv_counts1 = recv_counts_cpu1.tolist() # tensor to list
                tokens_received1 = sum(recv_counts1)
        ############################ END: TBO 1 ########################

        with nvtx.annotate("TBO-0", color='purple'):
            global_x = ops.all_to_all_wait(permutated_local_x, global_x, global_x_handle)


        ############################ TBO 1 ############################
        with nvtx.annotate("TBO-1", color='orange'):
            with nvtx.annotate("all2all", color='green'):
                # global_x1, global_x_handle1 = ops.all_to_all(
                permutated_local_x1, global_x1, global_x_handle1 = ops.all_to_all_async(
                    permutated_local_x1,
                    recv_counts1,
                    send_counts1,
                    group=self.ep_pg,
                    # async_op=True
                )
            
            with torch.no_grad():
                # this specifiyes for the received global tokens, which local expert they belong to
                global_x_local_expert_indices1 = torch.repeat_interleave(
                    global_x_local_expert_indices_0,
                    global_batch_size_per_local_expert1.flatten(),
                    output_size=tokens_received1,
                ) # e.g. [0, ...,  0, ... , 3, ..., 3, 0, ...] for 4 local experts
        ############################ END: TBO 1 ########################



        with nvtx.annotate("TBO-0", color='purple'):
            global_x = self._checkpointed_permute_routed_experts_unpermute(
                global_x=global_x,
                global_x_local_expert_indices=global_x_local_expert_indices,
                parallel_batch_size_per_local_expert_cpu=parallel_batch_size_per_local_expert_cpu
            )
            
        
                    
        ############################ TBO 1 ############################
        with nvtx.annotate("TBO-1", color='orange'):
            global_x1 = ops.all_to_all_wait(permutated_local_x1, global_x1, global_x_handle1)

        ############################ END: TBO 1 ########################
    
        with nvtx.annotate("TBO-0", color='purple'):

            ########## 8. reverse_all_to_all ###########

            with nvtx.annotate("reverse_all_to_all", color='green'):
                global_x = cast(torch.Tensor, global_x)
                global_x, local_x, local_x_handle = ops.all_to_all_async(
                # local_x, local_x_handle = ops.all_to_all(
                    global_x,
                    send_counts,
                    recv_counts,
                    group=self.ep_pg,
                    # async_op=True
                )


        ############################ TBO 1 ############################
        with nvtx.annotate("TBO-1", color='orange'):
            global_x1 = self._checkpointed_permute_routed_experts_unpermute(
                global_x=global_x1,
                global_x_local_expert_indices=global_x_local_expert_indices1,
                parallel_batch_size_per_local_expert_cpu=parallel_batch_size_per_local_expert_cpu1
            )

        ############################ END: TBO 1 ########################

        with nvtx.annotate("TBO-0", color='purple'):
            local_x = ops.all_to_all_wait(global_x, local_x, local_x_handle)

            # del global_x # done with global tokens
            ############################################ end
            
            
            ############ 9. Unpermute the (local) tokens returned by all-to-all communication ##########
            with nvtx.annotate("Unpermute-Merge local tokens", color='green'):
                local_x = moe_unpermute_no_compile(
                    inp=local_x,
                    row_id_map=reversed_local_x_permutation_mapping,
                    merging_probs=local_x_global_routed_expert_weights.view(-1, self.routed_experts_router.top_k),
                    restore_shape=hidden_shape_before_permute,
                    map_type='index',
                )
            ############################################ end
        
            
            local_x = local_x.view(in_shape)


            # weighted sum of the shared experts and routed experts
            if self.shared_experts is not None:
                assert mixed_shared_out is not None

                mlp_out = local_x + mixed_shared_out
            else:
                mlp_out = local_x

            final_out = attn_res_out + self.feed_forward_norm(mlp_out)

            #######################


        ############################ TBO 1 ############################
        with nvtx.annotate("TBO-1", color='orange'):
            x1_ctx = {
                "global_x1": global_x1,
                "send_counts1": send_counts1,
                "recv_counts1": recv_counts1,
                # "tokens_received1": tokens_received1,
                "reversed_local_x_permutation_mapping1": reversed_local_x_permutation_mapping1,
                "local_x_global_routed_expert_weights1": local_x_global_routed_expert_weights1,
                "hidden_shape_before_permute1": hidden_shape_before_permute1,
                "in_shape1": in_shape1,
                "mixed_shared_out1": mixed_shared_out1,
                "attn_res_out1": attn_res_out1,
                "last_block": self,
            }



        ############################ END: TBO 1 ########################

        
        return (
            final_out,
            x1_ctx, 
        )

    def apply_compile(self):
        super().apply_compile()

        self.combined_forward_ep_tbo = torch.compile(self.combined_forward_ep_tbo) 


    
    def checkpointed_combined_forward_ep_tbo(
        self,
        x0: torch.Tensor,
        x1_ctx: Dict,
        x1_is_fresh: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        if self.checkpoint_combined_ep_tbo:
            out = checkpoint(
                self.combined_forward_ep_tbo,
                x0,
                x1_ctx,
                x1_is_fresh,
                loss_div_factor=loss_div_factor,
                use_reentrant=False,
                **kwargs,
            )
            return cast(Tuple[torch.Tensor, Dict], out)
        else:
            return self.combined_forward_ep_tbo(
                x0,
                x1_ctx,
                x1_is_fresh,
                loss_div_factor=loss_div_factor,
                **kwargs,
            )
        
