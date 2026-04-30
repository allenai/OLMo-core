import math
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank

if TYPE_CHECKING:
    from .block import MoEFusedV2TransformerBlock

try:
    import torch.distributed._symmetric_memory as _symm_mem
except ImportError:
    _symm_mem = None  # type: ignore[assignment]


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
class _NoSyncSymmTransientSlot:
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
    buffers: _NoSyncSymmBuffers
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
            raise ValueError(f"slot_idx must be in [0, {self.num_slots - 1}] (got {slot_idx})")
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


def get_ep_no_sync_group_name(block: "MoEFusedV2TransformerBlock") -> str:
    if not block.ep_no_sync:
        raise RuntimeError("EP no-sync is not enabled for this block")
    if block._ep_symm_group_name is None:
        raise RuntimeError(
            f"EP no-sync group is not initialized (block={block.block_idx}, ep_enabled={block.ep_enabled})"
        )
    return block._ep_symm_group_name


def ep_no_sync_slot_for_lane(block: "MoEFusedV2TransformerBlock", lane_id: int) -> int:
    if lane_id < 0:
        raise ValueError(f"lane_id must be >= 0 (got {lane_id})")
    base_slot = block._ep_no_sync_shared_slot
    if block._ep_no_sync_shared_pool is not None:
        return (base_slot + lane_id) % block._ep_no_sync_shared_pool.num_slots
    return base_slot + lane_id


def resolve_ep_no_sync_chunk_reorder_backend() -> str:
    backend = os.getenv("OLMO_MOE_CHUNK_REORDER_BACKEND", "cuda").lower()
    if backend == "auto":
        backend = "cuda"
    if backend not in ("cuda", "triton", "te"):
        backend = "cuda"
    return backend


def get_or_init_ep_no_sync_symm_tensor(
    block: "MoEFusedV2TransformerBlock",
    *,
    name: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if _symm_mem is None:
        raise RuntimeError("EP no-sync requires torch.distributed._symmetric_memory")
    if block.ep_pg is None:
        raise RuntimeError("EP process group is not initialized")

    cached = block._ep_no_sync_symm_cache.get(name)
    needs_realloc = (
        cached is None
        or tuple(cached.shape) != tuple(shape)
        or cached.dtype != dtype
        or cached.device != device
    )
    if needs_realloc:
        try:
            symm_tensor = _symm_mem.empty(shape, dtype=dtype, device=device)
            _symm_mem.rendezvous(symm_tensor, group=block.ep_pg)
        except Exception as e:
            raise RuntimeError(
                f"Failed to allocate/rendezvous symmetric tensor '{name}' with shape={shape}, "
                f"dtype={dtype}, device={device}, block={block.block_idx}, rank={get_rank(block.ep_pg)}: {e}"
            ) from e
        block._ep_no_sync_symm_cache[name] = symm_tensor

    return block._ep_no_sync_symm_cache[name]


@torch.compiler.disable
def get_ep_no_sync_buffers(
    block: "MoEFusedV2TransformerBlock",
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
    assert block.routed_experts_router is not None

    ep_world_size = block.ep_world_size
    transient_slot: Optional[_NoSyncSymmTransientSlot] = None
    resolved_slot_idx = block._ep_no_sync_shared_slot if slot_idx is None else slot_idx
    name_suffix = f"_slot{resolved_slot_idx}" if slot_idx is not None else ""
    chunk_reorder_backend = resolve_ep_no_sync_chunk_reorder_backend()
    if block._ep_no_sync_shared_pool is not None:
        if chunk_reorder_backend == "te":
            if not block._ep_no_sync_te_backend_warned:
                warnings.warn(
                    "EP no-sync shared symm buffer reuse is disabled when "
                    "OLMO_MOE_CHUNK_REORDER_BACKEND=te. "
                    "Falling back to per-block symmetric buffers for safety.",
                    stacklevel=2,
                )
                block._ep_no_sync_te_backend_warned = True
        else:
            transient_slot = block._ep_no_sync_shared_pool.get_slot(
                slot_idx=resolved_slot_idx,
                dispatch_in_cap=dispatch_in_cap,
                dispatch_out_cap=dispatch_out_cap,
                combine_in_cap=combine_in_cap,
                combine_out_cap=combine_out_cap,
                need_dispatch_in=need_dispatch_in,
                need_dispatch_meta=need_dispatch_meta,
                include_dispatch_out=need_dispatch_out and block.ep_no_sync_share_dispatch_out,
                need_combine_in=need_combine_in,
                need_combine_meta=need_combine_meta,
                include_combine_out=need_combine_out and block.ep_no_sync_share_combine_out,
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
            dispatch_in = get_or_init_ep_no_sync_symm_tensor(
                block,
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
            dispatch_in_rank_splits = get_or_init_ep_no_sync_symm_tensor(
                block,
                name=f"dispatch_in_rank_splits{name_suffix}",
                shape=(ep_world_size,),
                dtype=torch.int64,
                device=device,
            )
        if transient_slot is not None and transient_slot.dispatch_rank_splits_offsets is not None:
            dispatch_rank_splits_offsets = transient_slot.dispatch_rank_splits_offsets.detach()
        else:
            dispatch_rank_splits_offsets = get_or_init_ep_no_sync_symm_tensor(
                block,
                name=f"dispatch_rank_splits_offsets{name_suffix}",
                shape=(2, ep_world_size),
                dtype=torch.int64,
                device=device,
            )
        if (
            transient_slot is not None
            and transient_slot.dispatch_tmp_rank_splits_offsets is not None
        ):
            dispatch_tmp_rank_splits_offsets = (
                transient_slot.dispatch_tmp_rank_splits_offsets.detach()
            )
        else:
            dispatch_tmp_rank_splits_offsets = get_or_init_ep_no_sync_symm_tensor(
                block,
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
            combine_in = get_or_init_ep_no_sync_symm_tensor(
                block,
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
            combine_in_rank_splits = get_or_init_ep_no_sync_symm_tensor(
                block,
                name=f"combine_in_rank_splits{name_suffix}",
                shape=(ep_world_size,),
                dtype=torch.int64,
                device=device,
            )
        if transient_slot is not None and transient_slot.combine_rank_splits_offsets is not None:
            combine_rank_splits_offsets = transient_slot.combine_rank_splits_offsets.detach()
        else:
            combine_rank_splits_offsets = get_or_init_ep_no_sync_symm_tensor(
                block,
                name=f"combine_rank_splits_offsets{name_suffix}",
                shape=(2, ep_world_size),
                dtype=torch.int64,
                device=device,
            )
        if (
            transient_slot is not None
            and transient_slot.combine_tmp_rank_splits_offsets is not None
        ):
            combine_tmp_rank_splits_offsets = (
                transient_slot.combine_tmp_rank_splits_offsets.detach()
            )
        else:
            combine_tmp_rank_splits_offsets = get_or_init_ep_no_sync_symm_tensor(
                block,
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
            dispatch_out = get_or_init_ep_no_sync_symm_tensor(
                block,
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
            combine_out = get_or_init_ep_no_sync_symm_tensor(
                block,
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


def iter_ep_no_sync_symm_tensors(block: "MoEFusedV2TransformerBlock") -> Iterator[torch.Tensor]:
    for tensor in block._ep_no_sync_symm_cache.values():
        if isinstance(tensor, torch.Tensor):
            yield tensor
    if block._ep_no_sync_shared_pool is not None:
        yield from block._ep_no_sync_shared_pool.iter_tensors()


def compute_ep_no_sync_rank_capacity(
    block: "MoEFusedV2TransformerBlock", num_out_tokens: int
) -> int:
    # `num_out_tokens` is the local routed-token count before EP dispatch.
    # Under balanced routing, the average received tokens per EP rank is this
    # same value (not divided by ep_world_size).
    return max(
        1,
        int(math.ceil(block.ep_no_sync_capacity_factor * float(num_out_tokens))),
    )
