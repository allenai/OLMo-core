from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

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
