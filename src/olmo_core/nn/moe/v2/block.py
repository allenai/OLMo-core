import math
import os
import threading
import warnings
import weakref
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

# Process-local cache for the EP->group "0" symmetric-memory alias.
# This makes repeated calls from multiple MoE blocks idempotent when they use
# the same EP group.
_EP_SYMM_GROUP0_ALIAS_LOCK = threading.Lock()
_EP_SYMM_GROUP0_ALIAS_RANKS: Optional[Tuple[int, ...]] = None

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.utils import barrier, get_fs_local_rank, get_rank, get_world_size
from olmo_core.ops import moe as ops
from olmo_core.ops import attach_auxiliary_loss
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.kernels import symm_mem_vdev2d as symm_mem_vdev2d_kernels
from olmo_core.kernels import (
    ScaledGroupedMMPrequantizedRHS,
    prequantize_scaled_grouped_mm_rhs,
    scaled_grouped_mm_q,
)
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
from ...moe.utils import (
    async_copy_to_cpu,
    record_stream_event_no_compile,
    wait_event_no_compile,
    wait_stream_no_compile,
)
from .routed_experts import RoutedExperts, RoutedExpertsConfig, requires_host_side_split_sizes, use_torch_grouped_mm
from .router import MoERouterConfigV2, MoERouterV2
from .shared_experts import SharedExperts
from .shared_experts import SharedExpertsConfig
from .fp8 import MoERowwiseFP8Config, normalize_rowwise_fp8_config
from olmo_core.kernels.mxfp8_utils import (
    dequantize_rows_from_mxfp8,
    dot_gathered_rows_mxfp8_with_grad,
    quantize_rows_to_mxfp8,
)

# backend: transformer_engine
from ..utils import (
    build_chunk_te_routing_map,
    moe_chunk_reorder_no_compile,
    moe_permute_1d_fused_drop_no_compile,
    moe_unpermute_1d_fused_drop_no_compile,
    moe_unpermute_no_compile,
    moe_permute_no_compile,
    moe_sort_chunks_by_index_no_compile,
)
from olmo_core.nn.transformer.config import (
    TransformerBlockConfig,
    TransformerBlockType,
)


class DebugNodeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name: str = "", print_forward: bool = True, print_backward: bool = True):
        # Save metadata (NOT a Tensor) in ctx
        ctx.name = name
        ctx.print_backward = print_backward

        # Save something from forward if you want to see it in backward
        # (be careful: saving big tensors increases memory)
        ctx.x_shape = tuple(x.shape)
        ctx.x_dtype = x.dtype
        ctx.x_device = x.device

        if print_forward:
            with torch.no_grad():
                print(f"[DebugNode fwd] {name} shape={ctx.x_shape} dtype={ctx.x_dtype} device={ctx.x_device} "
                      f"requires_grad={x.requires_grad}")

        # Identity
        return x

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.print_backward:
            with torch.no_grad():
                # grad_out is the gradient flowing into this node
                gshape = tuple(grad_out.shape) if isinstance(grad_out, torch.Tensor) else None
                gnorm = grad_out.norm().item() if isinstance(grad_out, torch.Tensor) else None
                print(f"[DebugNode bwd] {ctx.name} grad_shape={gshape} grad_norm={gnorm} "
                      f"(saved x_shape={ctx.x_shape} dtype={ctx.x_dtype} device={ctx.x_device})")

        # Gradient w.r.t. x is just grad_out; no grads for non-tensor args
        return grad_out, None, None, None


def debugNode(x, name="", print_forward=True, print_backward=True):
    return DebugNodeFn.apply(x, name, print_forward, print_backward)

@dataclass
class _NoSyncSymmBuffers:
    dispatch_in: torch.Tensor
    dispatch_in_rank_splits: torch.Tensor
    dispatch_out: torch.Tensor
    dispatch_out_is_shared: bool
    dispatch_rank_splits_offsets: torch.Tensor
    dispatch_tmp_rank_splits_offsets: torch.Tensor
    combine_in: torch.Tensor
    combine_in_rank_splits: torch.Tensor
    combine_out: torch.Tensor
    combine_out_is_shared: bool
    combine_rank_splits_offsets: torch.Tensor
    combine_tmp_rank_splits_offsets: torch.Tensor


@dataclass
class _NoSyncSymmTransientSlot:# shared in pool
    dispatch_in: Optional[torch.Tensor]
    dispatch_out: Optional[torch.Tensor]
    dispatch_in_rank_splits: Optional[torch.Tensor]
    dispatch_rank_splits_offsets: Optional[torch.Tensor]
    dispatch_tmp_rank_splits_offsets: Optional[torch.Tensor]
    combine_in: Optional[torch.Tensor]
    combine_out: Optional[torch.Tensor]
    combine_in_rank_splits: Optional[torch.Tensor]
    combine_rank_splits_offsets: Optional[torch.Tensor]
    combine_tmp_rank_splits_offsets: Optional[torch.Tensor]


@dataclass
class _NoSyncStageAState:
    lane_id: int
    slot_idx: int
    group_name: str
    in_shape: torch.Size
    hidden_shape_before_permute: torch.Size
    B: int
    S: int
    D: int
    attn_res_out: torch.Tensor
    mixed_shared_out: Optional[torch.Tensor]
    shared_done_event: Optional[torch.cuda.Event]
    local_x_global_routed_expert_weights: torch.Tensor
    routed_expert_router_aux_loss_info: Optional[Tuple[object, ...]]
    requested_splits: torch.Tensor
    allowed_splits: torch.Tensor
    recv_splits_by_src_local: torch.Tensor
    local_inverse_reorder_indices: torch.Tensor
    packed_keep_mask: torch.Tensor
    num_kept: torch.Tensor
    num_out_tokens: int
    send_rank_splits: torch.Tensor
    buffers: "_NoSyncSymmBuffers"
    permutated_local_x: torch.Tensor
    reversed_local_x_permutation_mapping: torch.Tensor


@dataclass
class _NoSyncStageDState:
    lane_id: int
    a_state: _NoSyncStageAState
    dispatch_out: torch.Tensor
    dispatch_rank_splits_offsets: torch.Tensor
    dispatch_done_event: torch.cuda.Event


@dataclass
class _NoSyncTboPendingContext:
    block: "MoEFusedV2TransformerBlock"
    lane_id: int
    a_state: _NoSyncStageAState
    dispatch_rank_splits_offsets: torch.Tensor
    global_x_rank_major: torch.Tensor
    combine_out: Optional[torch.Tensor] = None
    combine_done_event: Optional[torch.cuda.Event] = None


class _NoSyncSymmSharedPool:
    def __init__(self, *, num_slots: int, group: dist.ProcessGroup):
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1 (got {num_slots})")
        self.num_slots = num_slots
        self.group = group
        self._slot_caches: List[Dict[str, torch.Tensor]] = [{} for _ in range(num_slots)]

    def _get_or_init_slot_tensor(
        self,
        *,
        slot_idx: int,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if _symm_mem is None:
            raise RuntimeError("EP no-sync requires torch.distributed._symmetric_memory")

        slot_cache = self._slot_caches[slot_idx]
        cached = slot_cache.get(name)
        needs_realloc = (
            cached is None
            or tuple(cached.shape) != tuple(shape)
            or cached.dtype != dtype
            or cached.device != device
        )
        if needs_realloc:
            symm_tensor = _symm_mem.empty(shape, dtype=dtype, device=device)
            _symm_mem.rendezvous(symm_tensor, group=self.group)
            slot_cache[name] = symm_tensor
        return slot_cache[name]

    def get_slot(
        self,
        *,
        slot_idx: int,
        dispatch_in_cap: int,
        dispatch_out_cap: int,
        combine_in_cap: int,
        combine_out_cap: int,
        need_dispatch_in: bool,
        need_dispatch_meta: bool,
        include_dispatch_out: bool,
        need_combine_in: bool,
        need_combine_meta: bool,
        include_combine_out: bool,
        d_model: int,
        dtype: torch.dtype,
        device: torch.device,
        ep_world_size: int,
    ) -> _NoSyncSymmTransientSlot:
        if slot_idx < 0 or slot_idx >= self.num_slots:
            raise ValueError(
                f"slot_idx must be in [0, {self.num_slots - 1}] (got {slot_idx})"
            )
        if need_dispatch_in:
            dispatch_in = self._get_or_init_slot_tensor(
                slot_idx=slot_idx,
                name="dispatch_in",
                shape=(dispatch_in_cap, d_model),
                dtype=dtype,
                device=device,
            )
        else:
            dispatch_in = None
        if need_dispatch_meta:
            dispatch_in_rank_splits = self._get_or_init_slot_tensor(
                slot_idx=slot_idx,
                name="dispatch_in_rank_splits",
                shape=(ep_world_size,),
                dtype=torch.int64,
                device=device,
            )
        else:
            dispatch_in_rank_splits = None
        if include_dispatch_out:
            dispatch_out = self._get_or_init_slot_tensor(
                slot_idx=slot_idx,
                name="dispatch_out",
                shape=(dispatch_out_cap, d_model),
                dtype=dtype,
                device=device,
            )
        else:
            dispatch_out = None
        if need_dispatch_meta:
            dispatch_rank_splits_offsets = self._get_or_init_slot_tensor(
                slot_idx=slot_idx,
                name="dispatch_rank_splits_offsets",
                shape=(2, ep_world_size),
                dtype=torch.int64,
                device=device,
            )
            dispatch_tmp_rank_splits_offsets = self._get_or_init_slot_tensor(
                slot_idx=slot_idx,
                name="dispatch_tmp_rank_splits_offsets",
                shape=(2, ep_world_size),
                dtype=torch.int64,
                device=device,
            )
        else:
            dispatch_rank_splits_offsets = None
            dispatch_tmp_rank_splits_offsets = None
        if need_combine_in:
            combine_in = self._get_or_init_slot_tensor(
                slot_idx=slot_idx,  
                name="combine_in",
                shape=(combine_in_cap, d_model),
                dtype=dtype,
                device=device,
            )
        else:
            combine_in = None
        if include_combine_out:
            combine_out = self._get_or_init_slot_tensor(
                slot_idx=slot_idx,
                name="combine_out",
                shape=(combine_out_cap, d_model),
                dtype=dtype,
                device=device,
            )
        else:
            combine_out = None
        if need_combine_meta:
            combine_in_rank_splits = self._get_or_init_slot_tensor(
                slot_idx=slot_idx,
                name="combine_in_rank_splits",
                shape=(ep_world_size,),
                dtype=torch.int64,
                device=device,
            )
            combine_rank_splits_offsets = self._get_or_init_slot_tensor(
                slot_idx=slot_idx,
                name="combine_rank_splits_offsets",
                shape=(2, ep_world_size),
                dtype=torch.int64,
                device=device,
            )
            combine_tmp_rank_splits_offsets = self._get_or_init_slot_tensor(
                slot_idx=slot_idx,
                name="combine_tmp_rank_splits_offsets",
                shape=(2, ep_world_size),
                dtype=torch.int64,
                device=device,
            )
        else:
            combine_in_rank_splits = None
            combine_rank_splits_offsets = None
            combine_tmp_rank_splits_offsets = None
        return _NoSyncSymmTransientSlot(
            dispatch_in=dispatch_in,
            dispatch_in_rank_splits=dispatch_in_rank_splits,
            dispatch_out=dispatch_out,
            dispatch_rank_splits_offsets=dispatch_rank_splits_offsets,
            dispatch_tmp_rank_splits_offsets=dispatch_tmp_rank_splits_offsets,
            combine_in=combine_in,
            combine_out=combine_out,
            combine_in_rank_splits=combine_in_rank_splits,
            combine_rank_splits_offsets=combine_rank_splits_offsets,
            combine_tmp_rank_splits_offsets=combine_tmp_rank_splits_offsets,
        )

    def iter_tensors(self):
        for slot_cache in self._slot_caches:
            for tensor in slot_cache.values():
                yield tensor

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
        # Mirror dispatch backward: vdev may only write a prefix when rank
        # capacity exceeds received rows. Zero tail rows to avoid stale data
        # from previous uses leaking into upstream gradients.
        # symm_grad_input.zero_()
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


class _CombineVDev2DOffsetAutograd(torch.autograd.Function):
    @staticmethod
    # @torch.compiler.disable
    def forward(  # type: ignore[override]
        ctx,
        input: torch.Tensor,
        symm_input: torch.Tensor,
        src_ranks: torch.Tensor,
        src_rows: torch.Tensor,
        group_name: str,
        group: dist.ProcessGroup,
        nblocks: int,
    ) -> torch.Tensor:
        input_rows = input.shape[0]
        if input_rows != symm_input.shape[0]:
            raise RuntimeError(
                f"combine2d input rows ({input_rows}) must equal symmetric combine input capacity ({symm_input.shape[0]})"
            )
        if nblocks < 0:
            raise RuntimeError(f"nblocks must be >= 0 (got {nblocks})")
        if src_ranks.ndim != 2 or src_rows.ndim != 2:
            raise RuntimeError(
                "src_ranks/src_rows must be rank-2 [R, 1], "
                f"got {tuple(src_ranks.shape)} and {tuple(src_rows.shape)}"
            )
        if src_ranks.shape != src_rows.shape:
            raise RuntimeError("src_ranks/src_rows shape mismatch")
        if src_ranks.shape[1] != 1:
            raise RuntimeError(f"src_ranks/src_rows second dim must be 1, got {src_ranks.shape[1]}")
        src_ranks_i64 = (
            src_ranks
            if src_ranks.dtype == torch.long
            else src_ranks.to(dtype=torch.long)
        )
        src_rows_i64 = (
            src_rows
            if src_rows.dtype == torch.long
            else src_rows.to(dtype=torch.long)
        )
        if not src_ranks_i64.is_contiguous():
            src_ranks_i64 = src_ranks_i64.contiguous()
        if not src_rows_i64.is_contiguous():
            src_rows_i64 = src_rows_i64.contiguous()

        input_aliases_symm_input = (
            input.untyped_storage().data_ptr() == symm_input.untyped_storage().data_ptr()
            and input.storage_offset() == symm_input.storage_offset()
            and tuple(input.shape) == tuple(symm_input.shape)
            and tuple(input.stride()) == tuple(symm_input.stride())
        )
        if not input_aliases_symm_input:
            symm_input.copy_(input)

        combine_out = torch.empty(
            (src_ranks_i64.shape[0], symm_input.shape[1]),
            device=symm_input.device,
            dtype=symm_input.dtype,
        )

        # The 2D-offset communication stage is executed with one-to-one rowwise
        # gather using packed route maps [R, 1].
        symm_mem_vdev2d_kernels.rowwise_gather_get(
            symm_input,
            combine_out,
            src_ranks_i64,
            src_rows_i64,
            group_name,
            nblocks=nblocks,
        )

        ctx.group = group
        ctx.group_name = group_name
        ctx.nblocks = int(nblocks)
        ctx.symm_input = symm_input
        ctx.save_for_backward(src_ranks_i64, src_rows_i64)
        return combine_out

    @staticmethod
    # @torch.compiler.disable
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        backward_dst_ranks, backward_dst_rows = ctx.saved_tensors
        if grad_out.shape[0] != backward_dst_ranks.shape[0]:
            raise RuntimeError(
                "combine2d backward grad rows must match src_ranks rows: "
                f"{grad_out.shape[0]} vs {backward_dst_ranks.shape[0]}"
            )
        symm_grad_input = ctx.symm_input
        if grad_out.shape[1] != symm_grad_input.shape[1]:
            raise RuntimeError(
                "combine2d backward grad hidden dim must match symm_input hidden dim: "
                f"{grad_out.shape[1]} vs {symm_grad_input.shape[1]}"
            )
        # Clear destination before remote puts to avoid stale rows.
        symm_grad_input.zero_()

        grad_out_contig = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
        symm_mem_vdev2d_kernels.rowwise_dispatch_put(
            grad_out_contig,
            symm_grad_input,
            backward_dst_ranks,
            backward_dst_rows,
            ctx.group_name,
            nblocks=ctx.nblocks,
        )

        return symm_grad_input, None, None, None, None, None, None


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
        symm_grad_out_hp = ctx.symm_out_hp

        grad_out_aliases = (
            grad_out.untyped_storage().data_ptr() == symm_grad_out_hp.untyped_storage().data_ptr()
            and grad_out.storage_offset() == symm_grad_out_hp.storage_offset()
            and tuple(grad_out.shape) == tuple(symm_grad_out_hp.shape)
            and tuple(grad_out.stride()) == tuple(symm_grad_out_hp.stride())
        )
        if not grad_out_aliases:
            symm_grad_out_hp.copy_(grad_out)

        grad_q, grad_scales = quantize_rows_to_mxfp8(
            symm_grad_out_hp,
            block_size=ctx.block_size,
        )
        ctx.symm_out_q.copy_(grad_q)
        ctx.symm_out_scales.copy_(grad_scales)

        grad_input = torch.empty(
            (dst_ranks.shape[0], symm_grad_out_hp.shape[1]),
            device=symm_grad_out_hp.device,
            dtype=symm_grad_out_hp.dtype,
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
    ep_no_sync_use_rowwise_all_to_all: bool = False
    ep_no_sync_rowwise_nblocks: int = 256
    ep_no_sync_share_dispatch_out: bool = False
    ep_no_sync_capacity_factor: float = 1.125
    ep_no_sync_shared_slots: int = 1
    ep_no_sync_share_combine_out: bool = False
    ep_no_sync_major_align: int = 1
    ep_no_sync_restore_unpermute_backend: str = "te_fused"
    rowwise_fp8: Optional[MoERowwiseFP8Config] = None
        
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
        ep_no_sync_use_2d_all_to_all: bool = False,
        ep_no_sync_use_rowwise_all_to_all: bool = False,
        ep_no_sync_rowwise_nblocks: int = 0,
        ep_no_sync_share_dispatch_out: bool = True,
        ep_no_sync_capacity_factor: float = 2.0,
        ep_no_sync_shared_slots: int = 2,
        ep_no_sync_share_combine_out: bool = False,
        ep_no_sync_major_align: int = 1,
        ep_no_sync_restore_unpermute_backend: str = "te_fused",
        rowwise_fp8: Optional[MoERowwiseFP8Config] = None,
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
        self.rowwise_fp8 = normalize_rowwise_fp8_config(rowwise_fp8)
        self._rowwise_fp8_checked = False
        self._shared_rowwise_fp8_up_prequant: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._shared_rowwise_fp8_down_prequant: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._shared_rowwise_fp8_weight_versions: Optional[Tuple[int, int]] = None

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
            routed_experts.rowwise_fp8 = normalize_rowwise_fp8_config(routed_experts.rowwise_fp8)
            if self.rowwise_fp8 is not None and routed_experts.rowwise_fp8 is None:
                routed_experts.rowwise_fp8 = self.rowwise_fp8
            self.routed_experts = routed_experts.build(init_device=init_device)
            owner_ref = weakref.ref(self)
            self.routed_experts.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
            self.routed_experts.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
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
            owner_ref = weakref.ref(self)
            self.shared_experts.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
            self.shared_experts.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
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
        self.ep_no_sync_use_rowwise_all_to_all = ep_no_sync_use_rowwise_all_to_all
        self.ep_no_sync_rowwise_nblocks = int(ep_no_sync_rowwise_nblocks)
        self.ep_no_sync_share_dispatch_out = ep_no_sync_share_dispatch_out
        if self.ep_no_sync_use_2d_all_to_all:
            raise OLMoConfigurationError(
                "ep_no_sync_use_2d_all_to_all=True is no longer supported: "
                "the 2D all_to_all path was removed due to correctness/performance issues."
            )
        self.ep_no_sync_capacity_factor = ep_no_sync_capacity_factor
        self.ep_no_sync_shared_slots = ep_no_sync_shared_slots
        self.ep_no_sync_share_combine_out = ep_no_sync_share_combine_out
        self.ep_no_sync_major_align = ep_no_sync_major_align
        self.ep_no_sync_restore_unpermute_backend = ep_no_sync_restore_unpermute_backend.lower()
        self._ep_symm_group_name: Optional[str] = None
        self._ep_no_sync_symm_cache: Dict[str, torch.Tensor] = {}
        self._ep_no_sync_last_debug: Dict[str, torch.Tensor] = {}
        self._ep_no_sync_shared_pool: Optional[_NoSyncSymmSharedPool] = None
        self._ep_no_sync_shared_slot: int = 0
        self._ep_no_sync_te_backend_warned: bool = False
        # self._ep_no_sync_forward_call_count: int = 0

        if self.ep_no_sync_capacity_factor <= 0:
            raise OLMoConfigurationError(
                f"ep_no_sync_capacity_factor must be > 0 (got {self.ep_no_sync_capacity_factor})"
            )
        if self.ep_no_sync_shared_slots < 1:
            raise OLMoConfigurationError(
                f"ep_no_sync_shared_slots must be >= 1 (got {self.ep_no_sync_shared_slots})"
            )
        if self.ep_no_sync_major_align < 1:
            raise OLMoConfigurationError(
                f"ep_no_sync_major_align must be >= 1 (got {self.ep_no_sync_major_align})"
            )
        if self.ep_no_sync_rowwise_nblocks < 0:
            raise OLMoConfigurationError(
                f"ep_no_sync_rowwise_nblocks must be >= 0 (got {self.ep_no_sync_rowwise_nblocks})"
            )
        if self.ep_no_sync_restore_unpermute_backend not in ("te_fused", "te_legacy", "cuda"):
            raise OLMoConfigurationError(
                "ep_no_sync_restore_unpermute_backend must be one of "
                "'te_fused'|'te_legacy'|'cuda' "
                f"(got {self.ep_no_sync_restore_unpermute_backend!r})"
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

    def get_ep_no_sync_comm_stream(self) -> torch.cuda.Stream:
        return get_or_init_stream(id=f"ep_no_sync_comm_block_{self.block_idx}", priority=0)

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
                "all_to_all_vdev, but NVSHMEM is not available in this "
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
                "all_to_all_vdev. Failed to switch backend to NVSHMEM "
                f"(current={current_backend}, after_error={backend_after}): {e}. "
                "Call torch.distributed._symmetric_memory.set_backend('NVSHMEM') "
                "before any symmetric-memory allocations."
            ) from e

    def _try_alias_ep_group_as_world_for_symm_mem(self) -> bool:
        """
        Try to alias EP group metadata as symmetric-memory group "0" so NVSHMEM
        allocator bootstrap follows EP group topology instead of WORLD.

        Returns True when aliasing is active (or unnecessary because EP==WORLD),
        otherwise False so caller can fall back to WORLD bootstrap.
        """
        global _EP_SYMM_GROUP0_ALIAS_RANKS
        if _symm_mem is None or self.ep_pg is None:
            return False
        if dist.group.WORLD is None:
            return False
        if self.ep_pg.group_name == dist.group.WORLD.group_name:
            return True

        try:
            import torch.distributed.distributed_c10d as c10d
            from torch._C._distributed_c10d import _SymmetricMemory
        except Exception:
            return False

        try:
            alias_ranks = tuple(sorted(c10d._world.pg_group_ranks[self.ep_pg].keys()))
            with _EP_SYMM_GROUP0_ALIAS_LOCK:
                # Idempotent success: alias already installed for the same EP group.
                if _EP_SYMM_GROUP0_ALIAS_RANKS == alias_ranks:
                    return True

                # If group "0" is already registered by a different context,
                # do not overwrite global process state.
                if _symm_mem.is_symm_mem_enabled_for_group("0"):
                    return False

                global_ranks_str = "_".join(map(str, alias_ranks))
                store = c10d.PrefixStore(
                    f"symmetric_memory-{global_ranks_str}",
                    c10d._get_process_group_store(self.ep_pg),
                )
                _SymmetricMemory.set_group_info(
                    "0",
                    dist.get_rank(self.ep_pg),
                    dist.get_world_size(self.ep_pg),
                    store,
                )
                # Keep Python bookkeeping in sync to avoid duplicate registration.
                group_to_store = getattr(_symm_mem, "_group_name_to_store", None)
                if isinstance(group_to_store, dict):
                    group_to_store["0"] = store
                _EP_SYMM_GROUP0_ALIAS_RANKS = alias_ranks
                return True
        except Exception:
            return False

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        assert self.routed_experts is not None, "ep can only be applied when routed_experts is enabled"
        ep_dp_mesh = ep_mesh['ep_dp']
        ep_mp_mesh = ep_mesh['ep_mp']
        ep_pg = kwargs.get("ep_pg")
        self.ep_mesh = ep_mesh
        self.routed_experts.apply_ep(
            ep_mesh
        )
        owner_ref = weakref.ref(self)
        self.routed_experts.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self.routed_experts.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self.invalidate_rowwise_fp8_cache()
        self.num_local_routed_experts = self.routed_experts.num_local_experts
        self._ep_enabled = True
        self.ep_pg = ep_pg if ep_pg is not None else ep_mp_mesh.get_group()

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
            alias_group0_active = False
            try:
                _symm_mem.enable_symm_mem_for_group(group_name)
                # Default path: alias EP group as group "0" so NVSHMEM allocator
                # bootstrap tracks EP topology. This keeps 2x8 intra-node teams
                # from inheriting WORLD inter-node behavior.
                alias_group0_active = self._try_alias_ep_group_as_world_for_symm_mem()
                if not alias_group0_active:
                    # Fallback path for environments where aliasing private APIs
                    # are unavailable or group "0" is already occupied.
                    # _symm_mem.enable_symm_mem_for_group(world_group_name)
                    # Option: hard fail
                    raise RuntimeError(
                        f"Failed to alias EP group '{group_name}' as group '0' for symmetric memory support"
                    )

            except Exception as e:
                raise RuntimeError(
                    f"Failed to enable symmetric memory for EP group '{group_name}' "
                    f"(world='{world_group_name}', alias_group0_active={alias_group0_active}) "
                    f"(block={self.block_idx}, rank={get_rank(self.ep_pg)}): {e}"
                ) from e
            self._ep_symm_group_name = group_name
            self._ep_no_sync_symm_cache.clear()
            self._ep_no_sync_shared_pool = None
            self._ep_no_sync_shared_slot = 0
            self._ep_no_sync_te_backend_warned = False

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
        self.combined_forward_ep_tbo = torch.compile(self.combined_forward_ep_tbo) 
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

    def invalidate_rowwise_fp8_cache(self) -> None:
        if self.routed_experts is not None:
            self.routed_experts.invalidate_rowwise_fp8_cache()
        self._shared_rowwise_fp8_up_prequant = None
        self._shared_rowwise_fp8_down_prequant = None
        self._shared_rowwise_fp8_weight_versions = None

    @torch.no_grad()
    def refresh_rowwise_fp8_cache(self) -> None:
        cfg = self.rowwise_fp8
        if cfg is None or not cfg.enabled:
            self.invalidate_rowwise_fp8_cache()
            return
        if self.routed_experts is not None:
            self.routed_experts.refresh_rowwise_fp8_cache()
        if self.shared_experts is None:
            self._shared_rowwise_fp8_up_prequant = None
            self._shared_rowwise_fp8_down_prequant = None
            self._shared_rowwise_fp8_weight_versions = None
            return
        if self.shared_experts.w_up_gate.device.type != "cuda":
            self._shared_rowwise_fp8_up_prequant = None
            self._shared_rowwise_fp8_down_prequant = None
            self._shared_rowwise_fp8_weight_versions = None
            return
        up_rhs = self.shared_experts.w_up_gate.unsqueeze(0)
        down_rhs = self.shared_experts.w_down
        self._shared_rowwise_fp8_up_prequant = prequantize_scaled_grouped_mm_rhs(up_rhs)
        self._shared_rowwise_fp8_down_prequant = prequantize_scaled_grouped_mm_rhs(down_rhs)
        self._shared_rowwise_fp8_weight_versions = (
            int(self.shared_experts.w_up_gate._version),
            int(self.shared_experts.w_down._version),
        )

    def _maybe_refresh_shared_rowwise_fp8_cache(self) -> None:
        cfg = self.rowwise_fp8
        if cfg is None or not cfg.enabled:
            return
        if self.shared_experts is None:
            return
        versions = (
            int(self.shared_experts.w_up_gate._version),
            int(self.shared_experts.w_down._version),
        )
        if (
            self._shared_rowwise_fp8_up_prequant is None
            or self._shared_rowwise_fp8_down_prequant is None
            or self._shared_rowwise_fp8_weight_versions != versions
        ):
            self.refresh_rowwise_fp8_cache()

    def _shared_experts_forward1_rowwise_fp8(
        self,
        x: torch.Tensor,
        *,
        use_fast_accum: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.shared_experts is not None
        B, S, D = x.shape
        E, H = self.shared_experts.num_experts, self.shared_experts.hidden_size
        BS = B * S

        self._maybe_refresh_shared_rowwise_fp8_cache()
        x2 = x.reshape(BS, D)
        offs = torch.tensor([BS], device=x.device, dtype=torch.int32)
        up_gate = scaled_grouped_mm_q(
            x2,
            self.shared_experts.w_up_gate.unsqueeze(0),
            offs=offs,
            use_fast_accum=use_fast_accum,
            prequantized_rhs=self._shared_rowwise_fp8_up_prequant,
        )
        up_gate = up_gate.view(BS, E, 2, H).permute(1, 0, 2, 3)
        up, gate = up_gate.unbind(dim=2)
        return up, gate

    def _shared_experts_forward2_rowwise_fp8(
        self,
        up: torch.Tensor,
        gate: torch.Tensor,
        xshape: torch.Size,
        *,
        use_fast_accum: bool,
    ) -> torch.Tensor:
        assert self.shared_experts is not None
        E, _H = self.shared_experts.num_experts, self.shared_experts.hidden_size
        B, S, D = xshape
        BS = B * S

        gate = torch.nn.functional.silu(gate)
        hidden = up * gate

        self._maybe_refresh_shared_rowwise_fp8_cache()
        hidden_2d = hidden.reshape(E * BS, -1)
        offs = torch.arange(BS, E * BS + 1, BS, device=hidden.device, dtype=torch.int32)
        out_2d = scaled_grouped_mm_q(
            hidden_2d,
            self.shared_experts.w_down,
            offs=offs,
            use_fast_accum=use_fast_accum,
            prequantized_rhs=self._shared_rowwise_fp8_down_prequant,
        )
        return out_2d.view(E, BS, D).view(E, B, S, D)

    def _get_ep_no_sync_group_name(self) -> str:
        if not self.ep_no_sync:
            raise RuntimeError("EP no-sync is not enabled for this block")
        if self._ep_symm_group_name is None:
            raise RuntimeError(
                f"EP no-sync group is not initialized (block={self.block_idx}, ep_enabled={self.ep_enabled})"
            )
        return self._ep_symm_group_name

    def _ep_no_sync_slot_for_lane(self, lane_id: int) -> int:
        if lane_id < 0:
            raise ValueError(f"lane_id must be >= 0 (got {lane_id})")
        base_slot = self._ep_no_sync_shared_slot
        if self._ep_no_sync_shared_pool is not None:
            return (base_slot + lane_id) % self._ep_no_sync_shared_pool.num_slots
        return base_slot + lane_id

    def _resolve_ep_no_sync_chunk_reorder_backend(self) -> str:
        backend = os.getenv("OLMO_MOE_CHUNK_REORDER_BACKEND", "cuda").lower()
        if backend == "auto":
            backend = "cuda"
        if backend not in ("cuda", "triton", "te"):
            backend = "cuda"
        return backend

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

    @torch.compiler.disable # to reduce Dynamo/AOT overhead
    def _get_ep_no_sync_buffers(
        self,
        *,
        dispatch_in_cap: int,
        dispatch_out_cap: int,
        combine_in_cap: int,
        combine_out_cap: int,
        d_model: int,
        dtype: torch.dtype,
        device: torch.device,
        slot_idx: Optional[int] = None,
        need_dispatch_in: bool = True,
        need_dispatch_meta: bool = True,
        need_dispatch_out: bool = True,
        need_combine_in: bool = True,
        need_combine_meta: bool = True,
        need_combine_out: bool = True,
    ) -> _NoSyncSymmBuffers:
        assert self.routed_experts_router is not None

        ep_world_size = self.ep_world_size
        transient_slot: Optional[_NoSyncSymmTransientSlot] = None
        resolved_slot_idx = self._ep_no_sync_shared_slot if slot_idx is None else slot_idx
        name_suffix = f"_slot{resolved_slot_idx}" if slot_idx is not None else ""
        chunk_reorder_backend = self._resolve_ep_no_sync_chunk_reorder_backend()
        if self._ep_no_sync_shared_pool is not None:
            if chunk_reorder_backend == "te":
                if not self._ep_no_sync_te_backend_warned:
                    warnings.warn(
                        "EP no-sync shared symm buffer reuse is disabled when "
                        "OLMO_MOE_CHUNK_REORDER_BACKEND=te. "
                        "Falling back to per-block symmetric buffers for safety.",
                        stacklevel=2,
                    )
                    self._ep_no_sync_te_backend_warned = True
            else:
                transient_slot = self._ep_no_sync_shared_pool.get_slot(
                    slot_idx=resolved_slot_idx,
                    dispatch_in_cap=dispatch_in_cap,
                    dispatch_out_cap=dispatch_out_cap,
                    combine_in_cap=combine_in_cap,
                    combine_out_cap=combine_out_cap,
                    need_dispatch_in=need_dispatch_in,
                    need_dispatch_meta=need_dispatch_meta,
                    include_dispatch_out=need_dispatch_out and self.ep_no_sync_share_dispatch_out,
                    need_combine_in=need_combine_in,
                    need_combine_meta=need_combine_meta,
                    include_combine_out=need_combine_out and self.ep_no_sync_share_combine_out,
                    d_model=d_model,
                    dtype=dtype,
                    device=device,
                    ep_world_size=ep_world_size,
                )
        empty_data = torch.empty((0,), dtype=dtype, device=device)
        empty_i64 = torch.empty((0,), dtype=torch.int64, device=device)

        # Use fresh aliases per call to keep Tensor object identity unique across
        # layers while still sharing the same underlying storage.
        if need_dispatch_in:
            if transient_slot is not None and transient_slot.dispatch_in is not None:
                dispatch_in = transient_slot.dispatch_in.detach()
            else:
                dispatch_in = self._get_or_init_ep_no_sync_symm_tensor(
                    name=f"dispatch_in{name_suffix}",
                    shape=(dispatch_in_cap, d_model),
                    dtype=dtype,
                    device=device,
                )
        else:
            dispatch_in = empty_data

        if need_dispatch_meta:
            if transient_slot is not None and transient_slot.dispatch_in_rank_splits is not None:
                dispatch_in_rank_splits = transient_slot.dispatch_in_rank_splits.detach()
            else:
                dispatch_in_rank_splits = self._get_or_init_ep_no_sync_symm_tensor(
                    name=f"dispatch_in_rank_splits{name_suffix}",
                    shape=(ep_world_size,),
                    dtype=torch.int64,
                    device=device,
                )
            if transient_slot is not None and transient_slot.dispatch_rank_splits_offsets is not None:
                dispatch_rank_splits_offsets = transient_slot.dispatch_rank_splits_offsets.detach()
            else:
                dispatch_rank_splits_offsets = self._get_or_init_ep_no_sync_symm_tensor(
                    name=f"dispatch_rank_splits_offsets{name_suffix}",
                    shape=(2, ep_world_size),
                    dtype=torch.int64,
                    device=device,
                )
            if transient_slot is not None and transient_slot.dispatch_tmp_rank_splits_offsets is not None:
                dispatch_tmp_rank_splits_offsets = transient_slot.dispatch_tmp_rank_splits_offsets.detach()
            else:
                dispatch_tmp_rank_splits_offsets = self._get_or_init_ep_no_sync_symm_tensor(
                    name=f"dispatch_tmp_rank_splits_offsets{name_suffix}",
                    shape=(2, ep_world_size),
                    dtype=torch.int64,
                    device=device,
                )
        else:
            dispatch_in_rank_splits = empty_i64
            dispatch_rank_splits_offsets = empty_i64
            dispatch_tmp_rank_splits_offsets = empty_i64

        if need_combine_in:
            if transient_slot is not None and transient_slot.combine_in is not None:
                combine_in = transient_slot.combine_in.detach()
            else:
                combine_in = self._get_or_init_ep_no_sync_symm_tensor(
                    name=f"combine_in{name_suffix}",
                    shape=(combine_in_cap, d_model),
                    dtype=dtype,
                    device=device,
                )
        else:
            combine_in = empty_data

        if need_combine_meta:
            if transient_slot is not None and transient_slot.combine_in_rank_splits is not None:
                combine_in_rank_splits = transient_slot.combine_in_rank_splits.detach()
            else:
                combine_in_rank_splits = self._get_or_init_ep_no_sync_symm_tensor(
                    name=f"combine_in_rank_splits{name_suffix}",
                    shape=(ep_world_size,),
                    dtype=torch.int64,
                    device=device,
                )
            if transient_slot is not None and transient_slot.combine_rank_splits_offsets is not None:
                combine_rank_splits_offsets = transient_slot.combine_rank_splits_offsets.detach()
            else:
                combine_rank_splits_offsets = self._get_or_init_ep_no_sync_symm_tensor(
                    name=f"combine_rank_splits_offsets{name_suffix}",
                    shape=(2, ep_world_size),
                    dtype=torch.int64,
                    device=device,
                )
            if transient_slot is not None and transient_slot.combine_tmp_rank_splits_offsets is not None:
                combine_tmp_rank_splits_offsets = transient_slot.combine_tmp_rank_splits_offsets.detach()
            else:
                combine_tmp_rank_splits_offsets = self._get_or_init_ep_no_sync_symm_tensor(
                    name=f"combine_tmp_rank_splits_offsets{name_suffix}",
                    shape=(2, ep_world_size),
                    dtype=torch.int64,
                    device=device,
                )
        else:
            combine_in_rank_splits = empty_i64
            combine_rank_splits_offsets = empty_i64
            combine_tmp_rank_splits_offsets = empty_i64

        if need_dispatch_out:
            shared_dispatch_out = transient_slot.dispatch_out if transient_slot is not None else None
            if shared_dispatch_out is not None:
                dispatch_out = shared_dispatch_out.detach()
                dispatch_out_is_shared = True
            else:
                # Per-block fallback: required when shared pool is unavailable or disabled.
                dispatch_out = self._get_or_init_ep_no_sync_symm_tensor(
                    name=f"dispatch_out{name_suffix}",
                    shape=(dispatch_out_cap, d_model),
                    dtype=dtype,
                    device=device,
                )
                dispatch_out_is_shared = False
        else:
            dispatch_out = empty_data
            dispatch_out_is_shared = False

        if need_combine_out:
            shared_combine_out = transient_slot.combine_out if transient_slot is not None else None
            if shared_combine_out is not None:
                combine_out = shared_combine_out.detach()
                combine_out_is_shared = True
            else:
                # Per-block fallback: required when shared pool is unavailable or disabled.
                combine_out = self._get_or_init_ep_no_sync_symm_tensor(
                    name=f"combine_out{name_suffix}",
                    shape=(combine_out_cap, d_model),
                    dtype=dtype,
                    device=device,
                )
                combine_out_is_shared = False
        else:
            combine_out = empty_data
            combine_out_is_shared = False

        return _NoSyncSymmBuffers(
            dispatch_in=dispatch_in,
            dispatch_in_rank_splits=dispatch_in_rank_splits,
            dispatch_out=dispatch_out,
            dispatch_out_is_shared=dispatch_out_is_shared,
            dispatch_rank_splits_offsets=dispatch_rank_splits_offsets,
            dispatch_tmp_rank_splits_offsets=dispatch_tmp_rank_splits_offsets,
            combine_in=combine_in,
            combine_in_rank_splits=combine_in_rank_splits,
            combine_out=combine_out,
            combine_out_is_shared=combine_out_is_shared,
            combine_rank_splits_offsets=combine_rank_splits_offsets,
            combine_tmp_rank_splits_offsets=combine_tmp_rank_splits_offsets,
        )

    def iter_ep_no_sync_symm_tensors(self):
        for tensor in self._ep_no_sync_symm_cache.values():
            if isinstance(tensor, torch.Tensor):
                yield tensor
        if self._ep_no_sync_shared_pool is not None:
            yield from self._ep_no_sync_shared_pool.iter_tensors()

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
        return_keep_matrix: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Tail-drop keep-split sync with a single all-gather.
        Each EP rank receives every rank's requested splits, then computes the
        same keep policy locally using the shared `rank_capacity`.
        """
        # TODO: simplify this
        assert self.num_local_routed_experts is not None
        requested = requested_splits.to(dtype=torch.long)
        expected_splits = self.ep_world_size * self.num_local_routed_experts
        if requested.numel() != expected_splits:
            raise RuntimeError(
                "requested_splits size mismatch: "
                f"got {requested.numel()}, expected {expected_splits}"
            )

        gathered_payload = torch.empty(
            expected_splits * self.ep_world_size,
            device=requested.device,
            dtype=requested.dtype,
        )
        dist.all_gather_into_tensor(
            gathered_payload,
            requested,
            group=self.ep_pg,
        )

        gathered_payload_2d = gathered_payload.view(self.ep_world_size, expected_splits)
        global_requested = gathered_payload_2d.view(
            self.ep_world_size, self.ep_world_size, self.num_local_routed_experts
        )

        # Flatten in local-expert-major then source-rank order for each destination rank.
        counts_flat = global_requested.permute(2, 0, 1).reshape(-1, self.ep_world_size)
        cumsum_counts = torch.cumsum(counts_flat, dim=0)
        kept_cumsum = torch.clamp(cumsum_counts, max=rank_capacity)
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

        # shape: (source_rank, local_expert)
        recv_splits_by_src_local = keep_from_src_dest_local[:, local_rank, :]
        send_side_drop_token_count = requested.sum() - allowed_splits.sum()
        # receive_side_drop_token_count = rank_capacity - recv_splits_by_src_local.sum()
        if return_keep_matrix:
            return (
                allowed_splits,
                recv_splits_by_src_local,
                send_side_drop_token_count,
                keep_from_src_dest_local,
            )
        return allowed_splits, recv_splits_by_src_local, send_side_drop_token_count


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

        requested_ends = torch.cumsum(requested, dim=0)

        max_expert_idx = requested.numel() - 1
        # safe_token_ids = torch.arange(num_out_tokens, device=keep.device, dtype=torch.long)
        safe_token_ids = token_ids
        expert_ids = torch.searchsorted(requested_ends, safe_token_ids, right=True).clamp_max(max_expert_idx)
        starts = requested_ends - requested
        pos_in_chunk = safe_token_ids - starts.index_select(0, expert_ids)
        # keep_mask = in_range & (pos_in_chunk < keep.index_select(0, expert_ids))
        keep_mask = pos_in_chunk < keep.index_select(0, expert_ids)

        # Stable partition: keep rows first, then dropped rows.
        keep_i64 = keep_mask.to(dtype=torch.long)
        drop_i64 = (~keep_mask).to(dtype=torch.long)
        keep_rank = torch.cumsum(keep_i64, dim=0) - 1 # current token's position among kept tokens
        drop_rank = torch.cumsum(drop_i64, dim=0) - 1 # current token's position among dropped tokens
        num_kept = keep_i64.sum(dtype=torch.long)
        packed_pos = torch.where(keep_mask, keep_rank, num_kept + drop_rank) # position in the packed order (kept tokens followed by dropped tokens)

        reorder_indices = torch.empty_like(token_ids)
        reorder_indices.scatter_(0, packed_pos, token_ids)

        inverse_reorder_indices = packed_pos

        packed_keep_mask = keep_mask.index_select(0, reorder_indices)
        return reorder_indices, inverse_reorder_indices, packed_keep_mask

    @nvtx.annotate("_build_rowwise_route_maps")
    def _build_rowwise_route_maps(
        self,
        *,
        routing_map: torch.Tensor,
        allowed_splits: torch.Tensor,
        keep_from_src_dest_local: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build per-route destination rank/row maps for row-wise dispatch.
        Routes dropped by tail-capacity are encoded as -1.
        """
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
    def _build_rowwise_combine_2d_route_to_packed(
        self,
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



    @nvtx.annotate("_restore_drop_unpermute_1d")
    def _restore_drop_unpermute_1d(
        self,
        *,
        combine_out: torch.Tensor,
        local_inverse_reorder_indices: torch.Tensor,
        packed_keep_mask: torch.Tensor,
        num_kept: torch.Tensor,
        reversed_local_x_permutation_mapping: torch.Tensor,
        local_x_global_routed_expert_weights: torch.Tensor,
        hidden_shape_before_permute: torch.Size,
        row_id_map_is_packed: bool = False,
        backward_grad_input_buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.routed_experts_router is not None
        merging_probs = local_x_global_routed_expert_weights.view(-1, self.routed_experts_router.top_k)
        backend = self.ep_no_sync_restore_unpermute_backend

        if backend == "te_fused":
            return cast(
                torch.Tensor,
                moe_unpermute_1d_fused_drop_no_compile(
                    inp=combine_out,
                    row_id_map=reversed_local_x_permutation_mapping,
                    local_inverse_reorder_indices=local_inverse_reorder_indices,
                    packed_keep_mask=packed_keep_mask,
                    merging_probs=merging_probs,
                    num_kept=num_kept,
                    row_id_map_is_packed=row_id_map_is_packed,
                    backward_grad_input_buffer=backward_grad_input_buffer,
                    map_type="index",
                ),
            )
        if backend == "te_legacy":
            raise RuntimeError('te_legacy deprecated')
            if row_id_map_is_packed:
                restored_local_x = combine_out
            else:
                with nvtx.annotate("RestoreDrop", color="green"):
                    restored_local_x = combine_out.index_select(0, local_inverse_reorder_indices)
                    restored_keep_mask = packed_keep_mask.index_select(0, local_inverse_reorder_indices)
                    restored_local_x = torch.where(
                        restored_keep_mask.unsqueeze(-1),
                        restored_local_x,
                        torch.zeros_like(restored_local_x),
                    )
            return cast(
                torch.Tensor,
                moe_unpermute_no_compile(
                    inp=restored_local_x,
                    row_id_map=reversed_local_x_permutation_mapping,
                    merging_probs=merging_probs,
                    restore_shape=hidden_shape_before_permute,
                    map_type="index",
                ),
            )
        if backend == "cuda":
            raise RuntimeError(
                "ep_no_sync_restore_unpermute_backend='cuda' is not implemented yet. "
                "TODO: add a custom CUDA path mirroring TE _moe_unpermute_index_map semantics."
            )
        raise RuntimeError(f"Unhandled ep_no_sync_restore_unpermute_backend: {backend}")



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
        """Forward with EP no-sync using 1D symmetric-memory all_to_all_vdev ops."""
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

        if self.ep_no_sync_use_rowwise_all_to_all:
            return self.combined_forward_ep_no_sync_rowwise(
                x,
                loss_div_factor=loss_div_factor,
                **kwargs,
            )

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

        num_out_tokens = local_x_global_routed_expert_indices.numel() # a constant = B*S*top_k, including tokens that will be dropped

        with torch.no_grad():
            with nvtx.annotate("ConfigCapacity", color="green"):
                requested_splits = local_batch_size_per_global_routed_expert.to(dtype=torch.long)
                rank_capacity = self._compute_ep_no_sync_rank_capacity(num_out_tokens)
                allowed_splits, recv_splits_by_src_local, _drop_token_cnt = cast(
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    self._sync_tail_drop_allowed_splits_single_a2a(
                        requested_splits,
                        rank_capacity=rank_capacity,
                    ),
                )
                local_reorder_indices, local_inverse_reorder_indices, packed_keep_mask = self._build_keep_reorder(
                    requested_splits=requested_splits,
                    keep_splits=allowed_splits,
                    num_out_tokens=num_out_tokens,
                )
                num_kept = allowed_splits.sum(dtype=torch.long) # number of non-dropped tokens that will be dispatched
                dispatch_in_cap = num_out_tokens
                dispatch_out_cap = rank_capacity
                combine_in_cap = rank_capacity
                combine_out_cap = num_out_tokens

        buffers = self._get_ep_no_sync_buffers(
            dispatch_in_cap=dispatch_in_cap,
            dispatch_out_cap=dispatch_out_cap,
            combine_in_cap=combine_in_cap,
            combine_out_cap=combine_out_cap,
            d_model=moe_inp.shape[-1],
            dtype=moe_inp.dtype,
            device=moe_inp.device,
        )

        routing_map = local_x_global_routed_expert_indices.view(
            -1, self.routed_experts_router.top_k
        ).int()
        hidden_shape_before_permute = moe_inp.shape

        with torch.no_grad():
            padded_batch_size_per_local_expert = recv_splits_by_src_local.sum(
                dim=0,
                dtype=torch.long,
            )

        assert local_reorder_indices is not None
        assert local_inverse_reorder_indices is not None
        assert packed_keep_mask is not None
        assert num_kept is not None

        with nvtx.annotate("Permute local tokens", color="green"):
            # 1D all-to-all: one-shot custom CUDA permute+drop.
            # Expert-major permutation with per-expert tail-drop written directly to packed layout.
            permutated_local_x, reversed_local_x_permutation_mapping = moe_permute_1d_fused_drop_no_compile(
                inp=moe_inp,
                routing_map=routing_map,
                num_out_tokens=num_out_tokens,
                reorder_indices=local_reorder_indices,
                inverse_reorder_indices=local_inverse_reorder_indices,
                requested_splits=requested_splits,
                keep_splits=allowed_splits,
                out=buffers.dispatch_in.detach(),
                map_type="index",
            )

        with torch.no_grad():
            send_rank_splits = allowed_splits.view(
                self.ep_world_size, self.num_local_routed_experts
            ).sum(dim=-1, dtype=torch.long)

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

        dispatch_out, dispatch_rank_splits_offsets = _DispatchVDevAutograd.apply(
            permutated_local_x,
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

        # rank-major to local-expert-major for expert forward
        with nvtx.annotate("Permute global tokens", color='green'):
            if self.routed_experts.num_local_experts == 1:
                # should verify: use clone() to avoid in-place modification?
                dispatch_rank_major = dispatch_rank_major.clone()
                global_chunk_row_id_map = None
            else:
                with torch.no_grad():
                    global_chunk_routing_map = build_chunk_te_routing_map(
                        recv_splits_by_src_local,
                        rows=dispatch_rank_major.shape[0],
                    ) # (cap_rows, ) mapping each row to a local expert id
                dispatch_rank_major, global_chunk_row_id_map = moe_chunk_reorder_no_compile(
                    dispatch_rank_major,
                    routing_map=global_chunk_routing_map,
                    num_out_tokens=dispatch_rank_major.shape[0],
                    backward_grad_input_buffer=buffers.dispatch_out.detach(), # the backward should directly write into the dispatch_out buffer to save memory and copy
                    # backward_grad_input_buffer=None # for testing
                )

        # expert forward
        dispatch_rank_major = self.routed_experts(
            dispatch_rank_major,
            padded_batch_size_per_local_expert,
        )

        with nvtx.annotate("Unpermute global tokens", color='green'):
            if self.routed_experts.num_local_experts == 1:
                global_x_rank_major = dispatch_rank_major  # skip unpermute if only one local expert
            else:
                assert global_chunk_row_id_map is not None
                global_x_rank_major = moe_chunk_reorder_no_compile(
                    inp=dispatch_rank_major,
                    row_id_map=global_chunk_row_id_map,
                    # Use a fresh detached alias each call:
                    # - writes directly into combine buffer storage (no extra copy),
                    # - avoids persisting autograd history on the reusable buffer tensor object.
                    out=buffers.combine_in.detach(),
                )
                # This should give same results, but the autograd does not work with out=... arguments
                # torch.index_select(dispatch_rank_major, 0, global_chunk_row_id_map, out=buffers.combine_in.detach()) # out=... arguments don't support automatic differentiation
                # or this
                # global_x_rank_major = torch.gather(
                #     dispatch_rank_major,
                #     dim=0,
                #     index=global_chunk_row_id_map.unsqueeze(-1).expand(-1, dispatch_rank_major.shape[-1]),
                #     out=buffers.combine_in.detach(),
                # )

        # set shared experts to wait until this point to maximize overlap with alltoall
        wait_stream_no_compile(
            this_stream=self.get_dense_stream(),
            other_stream=torch.cuda.current_stream(),
        ) # type: ignore

        combine_out, _combine_rank_splits_offsets = _CombineVDevAutograd.apply(
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

        with nvtx.annotate("Unpermute-Merge local tokens", color="green"):
            # Shared combine_out storage may be reused by later layers before this
            # layer's backward runs, so keep a per-call snapshot for autograd.
            combine_out_for_unpermute = combine_out.clone() if buffers.combine_out_is_shared else combine_out
            # combine_out does not have capacity pads, but still has dropped tokens at the tail
            local_x = self._restore_drop_unpermute_1d(
                combine_out=combine_out_for_unpermute,
                local_inverse_reorder_indices=local_inverse_reorder_indices,
                packed_keep_mask=packed_keep_mask,
                num_kept=num_kept,
                reversed_local_x_permutation_mapping=reversed_local_x_permutation_mapping,
                local_x_global_routed_expert_weights=local_x_global_routed_expert_weights,
                hidden_shape_before_permute=hidden_shape_before_permute,
                row_id_map_is_packed=True,
                backward_grad_input_buffer=buffers.combine_out.detach(), # the backward should directly write into the combine_out buffer to save memory and copy
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

        local_x = local_x.view(in_shape)
        wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

        if self.shared_experts is not None:
            assert mixed_shared_out is not None
            mlp_out = local_x + mixed_shared_out
        else:
            mlp_out = local_x

        final_out = attn_res_out + self.feed_forward_norm(mlp_out)

        if routed_expert_router_aux_loss_info is not None:
            # TODO;BUG: load balancing loss is calculated twice (or at least logged 2x as large in wandb).   
            routed_expert_router_aux_loss = self.routed_experts_router.compute_aux_loss(
                *routed_expert_router_aux_loss_info
            )
            final_out = attach_auxiliary_loss(final_out, routed_expert_router_aux_loss)

        return final_out

    def combined_forward_ep_no_sync_rowwise(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with EP no-sync using row-wise NVSHMEM dispatch/combine."""
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

        num_out_tokens = local_x_global_routed_expert_indices.numel() # a constant = B*S*top_k, including tokens that will be dropped

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
        
        # optionally add wait here if want to make shared experts wait for sync token alltoall to finish.
        # sometimes shared experts take all the SMs and block the alltoall
        # wait_stream_no_compile(
        #     this_stream=self.get_dense_stream(),
        #     other_stream=torch.cuda.current_stream(),
        # )

        with torch.no_grad():
            dst_ranks, dst_rows = self._build_rowwise_route_maps(
                routing_map=routing_map,
                allowed_splits=allowed_splits,
                keep_from_src_dest_local=keep_from_src_dest_local,
            )
            rowwise_nblocks = self.ep_no_sync_rowwise_nblocks

        # lauch shared experts
        if self.shared_experts is not None:
            # wait_stream_no_compile(
            #     this_stream=self.get_dense_stream(),
            #     other_stream=torch.cuda.current_stream(),
            # ) # type: ignore
            with torch.cuda.stream(self.get_dense_stream()):
                if use_rowwise_fp8:
                    assert rowwise_fp8_cfg is not None
                    shared_out_up, shared_out_gate = self._shared_experts_forward1_rowwise_fp8(
                        attn_res_out,
                        use_fast_accum=rowwise_fp8_cfg.use_fast_accum,
                    )
                else:
                    shared_out_up, shared_out_gate = self.shared_experts.forward1(attn_res_out)
        else:
            shared_out_up, shared_out_gate = None, None

        with nvtx.annotate("Rowwise Dispatch", color="green"):
            # _record_before("dispatch")
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
            # _record_after("dispatch")

        # Expert backward (grouped_mm) saves its input tensor. The rowwise
        # dispatch output aliases reusable symmetric buffers, so later
        # microbatches/layers can overwrite it before backward.
        # Snapshot to stable storage for autograd correctness.
        # if torch.is_grad_enabled() and buffers.dispatch_out_is_shared:
        #     dispatch_rank_major = dispatch_rank_major.clone()

        # expert forward

        dispatch_rank_major = self.routed_experts(
            dispatch_rank_major,
            padded_batch_size_per_local_expert,
            # Write expert output directly into the combine symmetric buffer.
            down_proj_out=buffers.combine_in.detach(),
            rowwise_fp8_down_proj_out_q=(combine_in_q if use_rowwise_fp8 else None),
            rowwise_fp8_down_proj_out_scales=(combine_in_scales if use_rowwise_fp8 else None),
            # Write grad(input) for expert up+gate directly into the rowwise
            # dispatch symmetric buffer so rowwise dispatch backward can consume
            # it without an extra copy.
            up_proj_input_grad_out=buffers.dispatch_out.detach(),
            use_rowwise_fp8=use_rowwise_fp8,
            rowwise_fp8_input_q=(dispatch_out_q if use_rowwise_fp8 else None),
            rowwise_fp8_input_scales=(dispatch_out_scales if use_rowwise_fp8 else None),
        )

        wait_stream_no_compile(
            this_stream=self.get_dense_stream(),
            other_stream=torch.cuda.current_stream(),
        ) # type: ignore

        with nvtx.annotate("Rowwise Combine Merge", color="green"):
            route_probs = local_x_global_routed_expert_weights.view(
                -1, self.routed_experts_router.top_k
            )
            # _record_before("combine")
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
            # _record_after("combine")

        # launch second half of the shared expert forward
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

        final_out = attn_res_out + self.feed_forward_norm(mlp_out)

        if routed_expert_router_aux_loss_info is not None:
            # TODO;BUG: load balancing loss is calculated twice (or at least logged 2x as large in wandb).   
            routed_expert_router_aux_loss = self.routed_experts_router.compute_aux_loss(
                *routed_expert_router_aux_loss_info
            )
            final_out = attach_auxiliary_loss(final_out, routed_expert_router_aux_loss)

        return final_out

    def _ep_no_sync_stage_a(
        self,
        x: torch.Tensor,
        *,
        lane_id: int,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> _NoSyncStageAState:
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
        if self.ep_no_sync_use_rowwise_all_to_all:
            raise RuntimeError(
                "ep_no_sync_use_rowwise_all_to_all=True is only implemented for "
                "combined_forward_ep_no_sync() (non-TBO path) right now."
            )

        group_name = self._get_ep_no_sync_group_name()
        slot_idx = self._ep_no_sync_slot_for_lane(lane_id)
        B, S, D = x.shape
        block_inp = x
        del x

        # Keep attention kwargs intact for the second lane call.
        attn_kwargs = dict(kwargs)
        with nvtx.annotate("A-AttnRouter", color="purple"):
            attn_res_out = self._checkpointed_res_norm_attn(block_inp, **attn_kwargs)
            (
                local_x_global_routed_expert_weights,
                local_x_global_routed_expert_indices,
                local_batch_size_per_global_routed_expert,
                routed_expert_router_aux_loss_info,
            ) = self.router_forward(
                router=self.routed_experts_router,
                local_x=attn_res_out,
                scores_only=False,
                loss_div_factor=loss_div_factor,
            )

        mixed_shared_out: Optional[torch.Tensor]
        shared_done_event: Optional[torch.cuda.Event] = None
        wait_stream_no_compile(
            this_stream=self.get_dense_stream(),
            other_stream=torch.cuda.current_stream(),
        )
        dense_stream = self.get_dense_stream()
        with torch.cuda.stream(dense_stream):
            if self.shared_experts_router:
                (
                    local_x_global_shared_expert_weights,
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

            if self.shared_experts is not None:
                # Count shared experts as stage-A work.
                # shared_out_up, shared_out_gate = self.shared_experts.forward1(attn_res_out)
                # shared_out = self.shared_experts.forward2(shared_out_up, shared_out_gate, attn_res_out.shape)
                shared_out = self.shared_experts(attn_res_out)  # (E_shared, B, S, D)
                if self.shared_experts_router:
                    assert local_x_global_shared_expert_weights is not None
                    _, _, E_s = local_x_global_shared_expert_weights.shape
                    mixed_shared_out = torch.bmm(
                        local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(B * S, 1, E_s),
                        shared_out.permute(1, 2, 0, 3).contiguous().view(B * S, E_s, D),
                    ).squeeze(1).view(B, S, D)
                else:
                    mixed_shared_out = shared_out.squeeze(0)
                shared_done_event = record_stream_event_no_compile(dense_stream)
            else:
                mixed_shared_out = None

        moe_inp = attn_res_out
        in_shape = moe_inp.size()
        moe_inp = moe_inp.view(-1, in_shape[-1])  # (B*S, D)
        hidden_shape_before_permute = moe_inp.shape
        num_out_tokens = local_x_global_routed_expert_indices.numel()

        with torch.no_grad():
            with nvtx.annotate("A-ConfigCapacity", color="green"):
                requested_splits = local_batch_size_per_global_routed_expert.to(dtype=torch.long)
                rank_capacity = self._compute_ep_no_sync_rank_capacity(num_out_tokens)
                allowed_splits, recv_splits_by_src_local, _drop_token_cnt = cast(
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    self._sync_tail_drop_allowed_splits_single_a2a(
                        requested_splits,
                        rank_capacity=rank_capacity,
                    ),
                )
                local_reorder_indices, local_inverse_reorder_indices, packed_keep_mask = self._build_keep_reorder(
                    requested_splits=requested_splits,
                    keep_splits=allowed_splits,
                    num_out_tokens=num_out_tokens,
                )
                num_kept = allowed_splits.sum(dtype=torch.long)

                dispatch_in_cap = num_out_tokens
                dispatch_out_cap = rank_capacity
                combine_in_cap = rank_capacity
                combine_out_cap = num_out_tokens

        buffers = self._get_ep_no_sync_buffers(
            dispatch_in_cap=dispatch_in_cap,
            dispatch_out_cap=dispatch_out_cap,
            combine_in_cap=combine_in_cap,
            combine_out_cap=combine_out_cap,
            d_model=moe_inp.shape[-1],
            dtype=moe_inp.dtype,
            device=moe_inp.device,
            slot_idx=slot_idx,
        )

        with nvtx.annotate("A-PermuteLocal", color="green"):
            routing_map = local_x_global_routed_expert_indices.view(
                -1, self.routed_experts_router.top_k
            ).int()
            permutated_local_x, reversed_local_x_permutation_mapping = moe_permute_1d_fused_drop_no_compile(
                inp=moe_inp,
                routing_map=routing_map,
                num_out_tokens=num_out_tokens,
                reorder_indices=local_reorder_indices,
                inverse_reorder_indices=local_inverse_reorder_indices,
                requested_splits=requested_splits,
                keep_splits=allowed_splits,
                out=buffers.dispatch_in.detach(),
                map_type="index",
            )

        with torch.no_grad():
            send_rank_splits = allowed_splits.view(
                self.ep_world_size, self.num_local_routed_experts
            ).sum(dim=-1, dtype=torch.long)

        return _NoSyncStageAState(
            lane_id=lane_id,
            slot_idx=slot_idx,
            group_name=group_name,
            in_shape=in_shape,
            hidden_shape_before_permute=hidden_shape_before_permute,
            B=B,
            S=S,
            D=D,
            attn_res_out=attn_res_out,
            mixed_shared_out=mixed_shared_out,
            shared_done_event=shared_done_event,
            local_x_global_routed_expert_weights=local_x_global_routed_expert_weights,
            routed_expert_router_aux_loss_info=routed_expert_router_aux_loss_info,
            requested_splits=requested_splits,
            allowed_splits=allowed_splits,
            recv_splits_by_src_local=recv_splits_by_src_local,
            local_inverse_reorder_indices=local_inverse_reorder_indices,
            packed_keep_mask=packed_keep_mask,
            num_kept=num_kept,
            num_out_tokens=num_out_tokens,
            send_rank_splits=send_rank_splits,
            buffers=buffers,
            permutated_local_x=permutated_local_x,
            reversed_local_x_permutation_mapping=reversed_local_x_permutation_mapping,
        )

    def _ep_no_sync_stage_d_launch(self, a_state: _NoSyncStageAState) -> _NoSyncStageDState:
        comm_stream = self.get_ep_no_sync_comm_stream()
        wait_stream_no_compile(this_stream=comm_stream, other_stream=torch.cuda.current_stream())

        with torch.cuda.stream(comm_stream):
            dispatch_out, dispatch_rank_splits_offsets = _DispatchVDevAutograd.apply(
                a_state.permutated_local_x,
                a_state.send_rank_splits,
                a_state.buffers.dispatch_in,
                a_state.buffers.dispatch_in_rank_splits,
                a_state.buffers.dispatch_out,
                a_state.buffers.dispatch_rank_splits_offsets,
                a_state.buffers.dispatch_tmp_rank_splits_offsets,
                a_state.group_name,
                self.ep_pg,
            )
            dispatch_done_event = record_stream_event_no_compile(comm_stream)

        return _NoSyncStageDState(
            lane_id=a_state.lane_id,
            a_state=a_state,
            dispatch_out=dispatch_out,
            dispatch_rank_splits_offsets=dispatch_rank_splits_offsets,
            dispatch_done_event=dispatch_done_event,
        )

    def _ep_no_sync_stage_e(self, d_state: _NoSyncStageDState) -> _NoSyncTboPendingContext:
        assert self.routed_experts is not None
        wait_event_no_compile(torch.cuda.current_stream(), d_state.dispatch_done_event)

        a_state = d_state.a_state
        buffers = a_state.buffers
        dispatch_rank_major = d_state.dispatch_out

        with torch.no_grad():
            padded_batch_size_per_local_expert = a_state.recv_splits_by_src_local.sum(
                dim=0,
                dtype=torch.long,
            )

        with nvtx.annotate("E-PermuteGlobal", color="green"):
            if self.routed_experts.num_local_experts == 1:
                dispatch_rank_major = dispatch_rank_major.clone()
                global_chunk_row_id_map = None
            else:
                with torch.no_grad():
                    global_chunk_routing_map = build_chunk_te_routing_map(
                        a_state.recv_splits_by_src_local,
                        rows=dispatch_rank_major.shape[0],
                    )
                dispatch_rank_major, global_chunk_row_id_map = moe_chunk_reorder_no_compile(
                    inp=dispatch_rank_major,
                    routing_map=global_chunk_routing_map,
                    num_out_tokens=dispatch_rank_major.shape[0],
                    backward_grad_input_buffer=buffers.dispatch_out.detach(),
                )

        with nvtx.annotate("E-RoutedExperts", color="green"):
            dispatch_rank_major = self.routed_experts(
                dispatch_rank_major,
                padded_batch_size_per_local_expert,
            )

        with nvtx.annotate("E-UnpermuteGlobal", color="green"):
            if self.routed_experts.num_local_experts == 1:
                global_x_rank_major = dispatch_rank_major
            else:
                assert global_chunk_row_id_map is not None
                global_x_rank_major = moe_chunk_reorder_no_compile(
                    inp=dispatch_rank_major,
                    row_id_map=global_chunk_row_id_map,
                    out=buffers.combine_in.detach(),
                )

        return _NoSyncTboPendingContext(
            block=self,
            lane_id=d_state.lane_id,
            a_state=a_state,
            dispatch_rank_splits_offsets=d_state.dispatch_rank_splits_offsets,
            global_x_rank_major=global_x_rank_major,
        )

    def _ep_no_sync_stage_c_launch(
        self, pending_ctx: _NoSyncTboPendingContext
    ) -> _NoSyncTboPendingContext:
        block = pending_ctx.block
        a_state = pending_ctx.a_state
        buffers = a_state.buffers
        comm_stream = block.get_ep_no_sync_comm_stream()
        wait_stream_no_compile(this_stream=comm_stream, other_stream=torch.cuda.current_stream())

        with torch.cuda.stream(comm_stream):
            combine_out, _combine_rank_splits_offsets = _CombineVDevAutograd.apply(
                pending_ctx.global_x_rank_major,
                pending_ctx.dispatch_rank_splits_offsets[0],
                buffers.combine_in,
                buffers.combine_in_rank_splits,
                buffers.combine_out,
                buffers.combine_rank_splits_offsets,
                buffers.combine_tmp_rank_splits_offsets,
                a_state.group_name,
                block.ep_pg,
            )
            combine_done_event = record_stream_event_no_compile(comm_stream)

        pending_ctx.combine_out = combine_out
        pending_ctx.combine_done_event = combine_done_event
        return pending_ctx

    def _ep_no_sync_stage_tail(self, pending_ctx: _NoSyncTboPendingContext) -> torch.Tensor:
        if pending_ctx.combine_done_event is not None:
            wait_event_no_compile(torch.cuda.current_stream(), pending_ctx.combine_done_event)

        a_state = pending_ctx.a_state
        buffers = a_state.buffers
        if a_state.shared_done_event is not None:
            wait_event_no_compile(torch.cuda.current_stream(), a_state.shared_done_event)

        combine_out = pending_ctx.combine_out if pending_ctx.combine_out is not None else buffers.combine_out
        with nvtx.annotate("Tail-UnpermuteMerge", color="green"):
            combine_out_for_unpermute = combine_out.clone() if buffers.combine_out_is_shared else combine_out
            local_x = self._restore_drop_unpermute_1d(
                combine_out=combine_out_for_unpermute,
                local_inverse_reorder_indices=a_state.local_inverse_reorder_indices,
                packed_keep_mask=a_state.packed_keep_mask,
                num_kept=a_state.num_kept,
                reversed_local_x_permutation_mapping=a_state.reversed_local_x_permutation_mapping,
                local_x_global_routed_expert_weights=a_state.local_x_global_routed_expert_weights,
                hidden_shape_before_permute=a_state.hidden_shape_before_permute,
                row_id_map_is_packed=True,
                backward_grad_input_buffer=buffers.combine_out.detach(),
            )

        local_x = local_x.view(a_state.in_shape)
        wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

        if self.shared_experts is not None:
            assert a_state.mixed_shared_out is not None
            mlp_out = local_x + a_state.mixed_shared_out
        else:
            mlp_out = local_x

        final_out = a_state.attn_res_out + self.feed_forward_norm(mlp_out)
        if a_state.routed_expert_router_aux_loss_info is not None:
            assert self.routed_experts_router is not None
            routed_expert_router_aux_loss = self.routed_experts_router.compute_aux_loss(
                *a_state.routed_expert_router_aux_loss_info
            )
            final_out = attach_auxiliary_loss(final_out, routed_expert_router_aux_loss)
        return final_out

    def combined_forward_ep_no_sync_tbo(
        self,
        x0: torch.Tensor,
        x1_ctx: object,
        x1_is_fresh: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, _NoSyncTboPendingContext]:
        if x1_is_fresh:
            pending_prev = None
        else:
            if not isinstance(x1_ctx, _NoSyncTboPendingContext):
                raise RuntimeError(
                    "Expected no-sync TBO context from previous block, "
                    f"got type={type(x1_ctx)}"
                )
            pending_prev = x1_ctx

        # A0 || C1(prev)
        with nvtx.annotate("TBO-1", color="orange"):
            if pending_prev is not None:
                pending_prev = pending_prev.block._ep_no_sync_stage_c_launch(pending_prev)
        with nvtx.annotate("TBO-0", color="purple"):
            a0 = self._ep_no_sync_stage_a(
                x0,
                lane_id=0,
                loss_div_factor=loss_div_factor,
                **kwargs,
            )

        # finish Tail1(prev) to materialize current block input lane-1
        with nvtx.annotate("TBO-1", color="orange"):
            if x1_is_fresh:
                fresh_ctx = cast(Dict[str, torch.Tensor], x1_ctx)
                block_inp1 = fresh_ctx["x1"]
            else:
                assert pending_prev is not None
                block_inp1 = pending_prev.block._ep_no_sync_stage_tail(pending_prev)

        # D0 || A1
        with nvtx.annotate("TBO-0", color="purple"):
            d0 = self._ep_no_sync_stage_d_launch(a0)
        with nvtx.annotate("TBO-1", color="orange"):
            a1 = self._ep_no_sync_stage_a(
                block_inp1,
                lane_id=1,
                loss_div_factor=loss_div_factor,
                **kwargs,
            )

        # E0 || D1
        with nvtx.annotate("TBO-1", color="orange"):
            d1 = self._ep_no_sync_stage_d_launch(a1)
        with nvtx.annotate("TBO-0", color="purple"):
            pending0_pre_c = self._ep_no_sync_stage_e(d0)

        # C0 || E1
        with nvtx.annotate("TBO-0", color="purple"):
            pending0_post_c = self._ep_no_sync_stage_c_launch(pending0_pre_c)
        with nvtx.annotate("TBO-1", color="orange"):
            pending1_pre_c = self._ep_no_sync_stage_e(d1)

        with nvtx.annotate("TBO-0", color="purple"):
            final_out = self._ep_no_sync_stage_tail(pending0_post_c)

        return final_out, pending1_pre_c

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

    def combined_forward_ep_tbo(
        self,
        x0: torch.Tensor,
        x1_ctx: object,
        x1_is_fresh: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, object]:
        if self.ep_no_sync:
            return self.combined_forward_ep_no_sync_tbo(
                x0,
                x1_ctx,
                x1_is_fresh,
                loss_div_factor=loss_div_factor,
                **kwargs,
            )

        x1_ctx = cast(Dict, x1_ctx)
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


    def checkpointed_combined_forward_ep_tbo(
        self,
        x0: torch.Tensor,
        x1_ctx: object,
        x1_is_fresh: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, object]:
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
            return cast(Tuple[torch.Tensor, object], out)
        else:
            return self.combined_forward_ep_tbo(
                x0,
                x1_ctx,
                x1_is_fresh,
                loss_div_factor=loss_div_factor,
                **kwargs,
            )
        
