import math
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank
from olmo_core.kernels import olmo_symm_mem

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock

try:
    import torch.distributed._symmetric_memory as _symm_mem
except ImportError:
    _symm_mem = None  # type: ignore[assignment]


def _tbo_buffer_debug_enabled() -> bool:
    if os.getenv("OLMO_TBO_VERBOSE_DEBUG_PRINT", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    ranks = os.getenv("OLMO_TBO_DEBUG_RANKS")
    if not ranks or not dist.is_available() or not dist.is_initialized():
        return True
    rank = str(dist.get_rank())
    return rank in {part.strip() for part in ranks.split(",") if part.strip()}


def _tbo_buffer_debug_print(message: str) -> None:
    if not _tbo_buffer_debug_enabled():
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else "?"
    print(
        (
            "[OLMO_TBO_DEBUG] "
            f"rank={rank} local_rank={os.getenv('LOCAL_RANK', '?')} "
            f"{message}"
        ),
        flush=True,
    )


def _cached_symm_tensor_covers(
    cached: torch.Tensor,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    cached_shape = tuple(cached.shape)
    return (
        len(cached_shape) == len(shape)
        and cached_shape[0] >= shape[0]
        and cached_shape[1:] == shape[1:]
        and cached.dtype == dtype
        and cached.device == device
    )


def _view_cached_symm_tensor(cached: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    if tuple(cached.shape) == shape:
        return cached
    view = cached.narrow(0, 0, shape[0])
    if not view.is_contiguous():
        raise RuntimeError(
            "Expected leading-dimension narrow of symmetric tensor to be contiguous: "
            f"{tuple(cached.shape)} -> {shape}"
        )
    return view


def _alloc_ep_symm_tensor(
    *,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    # _tbo_buffer_debug_print(f"symm_alloc:enter shape={shape} dtype={dtype} device={device}")
    if olmo_symm_mem.is_enabled():
        symm_tensor = olmo_symm_mem.empty(shape, dtype=dtype, device=device, group=group)
        olmo_symm_mem.rendezvous(symm_tensor, group=group)
        # _tbo_buffer_debug_print(f"symm_alloc:exit shape={shape} dtype={dtype} device={device}")
        return symm_tensor

    if _symm_mem is None:
        raise RuntimeError("EP no-sync requires torch.distributed._symmetric_memory")
    symm_tensor = _symm_mem.empty(shape, dtype=dtype, device=device)
    _symm_mem.rendezvous(symm_tensor, group=group)
    # _tbo_buffer_debug_print(f"symm_alloc:exit shape={shape} dtype={dtype} device={device}")
    return symm_tensor


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
    combine_gather: torch.Tensor
    combine_out_is_shared: bool
    combine_rank_splits_offsets: torch.Tensor
    combine_tmp_rank_splits_offsets: torch.Tensor
    dispatch_out_lease: Optional["_NoSyncSymmLease"] = None
    combine_out_lease: Optional["_NoSyncSymmLease"] = None
    combine_gather_lease: Optional["_NoSyncSymmLease"] = None


@dataclass
class _NoSyncRowwiseFP8SymmBuffers:
    dispatch_out_q: torch.Tensor
    dispatch_out_scales: torch.Tensor
    combine_in_q: torch.Tensor
    combine_in_scales: torch.Tensor
    dispatch_out_lease: Optional["_NoSyncSymmLease"] = None


@dataclass
class _NoSyncRowwiseLifetimeLeases:
    dispatch_out_lease: Optional["_NoSyncSymmLease"] = None
    combine_out_lease: Optional["_NoSyncSymmLease"] = None
    combine_gather_lease: Optional["_NoSyncSymmLease"] = None


@dataclass(frozen=True)
class _NoSyncSymmLeaseTensorSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class _NoSyncSymmLease:
    def __init__(
        self,
        *,
        pool: "_NoSyncSymmLeasePool",
        slot_idx: int,
        tensors: Dict[str, torch.Tensor],
    ):
        self._pool = pool
        self.slot_idx = slot_idx
        self._tensors = tensors
        self._released = False

    def tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]

    def release(self) -> None:
        if self._released:
            raise RuntimeError(
                f"Symmetric-memory lease for pool '{self._pool.name}' slot {self.slot_idx} "
                "was released more than once"
            )
        self._released = True
        self._pool.release(self.slot_idx)


class _NoSyncSymmLeasePool:
    def __init__(self, *, name: str, group: dist.ProcessGroup):
        self.name = name
        self.group = group
        self._slots: List[Dict[str, torch.Tensor]] = []
        self._free_slots: List[int] = []
        self._in_use_slots: set[int] = set()
        self.high_water: int = 0
        self.frozen: bool = False

    def _debug_enabled(self) -> bool:
        if os.getenv("OLMO_MOE_SYMM_LEASE_DEBUG", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
        ranks = os.getenv("OLMO_MOE_SYMM_LEASE_DEBUG_RANKS") or os.getenv("OLMO_TBO_DEBUG_RANKS")
        if not ranks or not dist.is_available() or not dist.is_initialized():
            return True
        rank = str(dist.get_rank())
        return rank in {part.strip() for part in ranks.split(",") if part.strip()}

    def _debug_print(self, message: str) -> None:
        if not self._debug_enabled():
            return
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else "?"
        print(
            (
                "[OLMO_MOE_SYMM_LEASE] "
                f"rank={rank} local_rank={os.getenv('LOCAL_RANK', '?')} "
                f"pool={self.name} {message}"
            ),
            flush=True,
        )

    @staticmethod
    def _spec_numel(shape: Tuple[int, ...]) -> int:
        numel = 1
        for dim in shape:
            numel *= int(dim)
        return numel

    @classmethod
    def _spec_bytes(cls, spec: _NoSyncSymmLeaseTensorSpec) -> int:
        return cls._spec_numel(spec.shape) * torch.empty((), dtype=spec.dtype).element_size()

    @classmethod
    def _format_specs(cls, specs: Tuple[_NoSyncSymmLeaseTensorSpec, ...]) -> str:
        return ", ".join(
            (
                f"{spec.name}:shape={spec.shape}:dtype={spec.dtype}:"
                f"bytes={cls._spec_bytes(spec)}"
            )
            for spec in specs
        )

    @classmethod
    def _specs_bytes(cls, specs: Tuple[_NoSyncSymmLeaseTensorSpec, ...]) -> int:
        return sum(cls._spec_bytes(spec) for spec in specs)

    def _dry_run_done(self) -> bool:
        try:
            from olmo_core.train.globals import get_global_arg

            return bool(get_global_arg("dry_run_done", default=False))
        except Exception:
            return False

    def _maybe_freeze_after_dry_run(self) -> None:
        if not self.frozen and self._dry_run_done():
            self.frozen = True
            # self._debug_print(
            #     f"freeze slots={len(self._slots)} high_water={self.high_water}"
            # )

    def _slot_covers(
        self,
        slot_idx: int,
        specs: Tuple[_NoSyncSymmLeaseTensorSpec, ...],
    ) -> bool:
        slot = self._slots[slot_idx]
        for spec in specs:
            cached = slot.get(spec.name)
            if cached is None or not _cached_symm_tensor_covers(
                cached,
                spec.shape,
                spec.dtype,
                spec.device,
            ):
                return False
        return True

    def _ensure_slot(
        self,
        slot_idx: int,
        specs: Tuple[_NoSyncSymmLeaseTensorSpec, ...],
    ) -> None:
        slot = self._slots[slot_idx]
        for spec in specs:
            cached = slot.get(spec.name)
            needs_realloc = cached is None or not _cached_symm_tensor_covers(
                cached,
                spec.shape,
                spec.dtype,
                spec.device,
            )
            if not needs_realloc:
                continue
            if self.frozen:
                raise RuntimeError(
                    f"Frozen symmetric-memory lease pool '{self.name}' slot {slot_idx} "
                    f"does not cover tensor '{spec.name}' with shape={spec.shape}, "
                    f"dtype={spec.dtype}, device={spec.device}"
                )
            # self._debug_print(
            #     f"alloc_begin slot={slot_idx} tensor={spec.name} "
            #     f"shape={spec.shape} dtype={spec.dtype} device={spec.device} "
            #     f"bytes={self._spec_bytes(spec)}"
            # )
            slot[spec.name] = _alloc_ep_symm_tensor(
                shape=spec.shape,
                dtype=spec.dtype,
                device=spec.device,
                group=self.group,
            )
            # self._debug_print(
            #     f"alloc_done slot={slot_idx} tensor={spec.name} "
            #     f"shape={spec.shape} dtype={spec.dtype} device={spec.device}"
            # )

    def _append_slot(self, specs: Tuple[_NoSyncSymmLeaseTensorSpec, ...]) -> int:
        slot_idx = len(self._slots)
        self._slots.append({})
        self._ensure_slot(slot_idx, specs)
        self._free_slots.append(slot_idx)
        return slot_idx

    def prewarm(
        self,
        *,
        num_slots: int,
        specs: Tuple[_NoSyncSymmLeaseTensorSpec, ...],
    ) -> None:
        if num_slots < 0:
            raise ValueError(f"num_slots must be >= 0 (got {num_slots})")
        self._maybe_freeze_after_dry_run()
        while len(self._slots) < num_slots:
            self._append_slot(specs)
        for slot_idx in range(len(self._slots)):
            if slot_idx in self._in_use_slots:
                continue
            self._ensure_slot(slot_idx, specs)
        bytes_per_slot = self._specs_bytes(specs)
        # self._debug_print(
        #     f"prewarm requested_slots={num_slots} slots={len(self._slots)} "
        #     f"free={len(self._free_slots)} in_use={len(self._in_use_slots)} "
        #     f"high_water={self.high_water} bytes_per_slot={bytes_per_slot} "
        #     f"total_bytes={bytes_per_slot * len(self._slots)} "
        #     f"specs=[{self._format_specs(specs)}]"
        # )

    def acquire(
        self,
        *,
        specs: Tuple[_NoSyncSymmLeaseTensorSpec, ...],
        owner: str,
    ) -> _NoSyncSymmLease:
        self._maybe_freeze_after_dry_run()
        slot_idx: Optional[int] = None
        skipped_slots: List[int] = []
        while self._free_slots:
            candidate = self._free_slots.pop()
            if self._slot_covers(candidate, specs) or not self.frozen:
                slot_idx = candidate
                break
            skipped_slots.append(candidate)
        self._free_slots.extend(skipped_slots)

        if slot_idx is None:
            if self.frozen:
                raise RuntimeError(
                    f"Frozen symmetric-memory lease pool '{self.name}' has no free slot "
                    f"for {owner}; slots={len(self._slots)} high_water={self.high_water} "
                    f"in_use={sorted(self._in_use_slots)}"
                )
            slot_idx = self._append_slot(specs)
            self._free_slots.remove(slot_idx)

        self._ensure_slot(slot_idx, specs)
        self._in_use_slots.add(slot_idx)
        self.high_water = max(self.high_water, len(self._in_use_slots))
        # self._debug_print(
        #     f"acquire owner={owner} slot={slot_idx} "
        #     f"in_use={len(self._in_use_slots)} high_water={self.high_water} "
        #     f"slots={len(self._slots)} free={len(self._free_slots)}"
        # )
        slot = self._slots[slot_idx]
        tensors = {
            spec.name: _view_cached_symm_tensor(slot[spec.name], spec.shape)
            for spec in specs
        }
        return _NoSyncSymmLease(pool=self, slot_idx=slot_idx, tensors=tensors)

    def release(self, slot_idx: int) -> None:
        if slot_idx not in self._in_use_slots:
            raise RuntimeError(
                f"Cannot release slot {slot_idx} from symmetric-memory lease pool "
                f"'{self.name}' because it is not in use"
            )
        self._in_use_slots.remove(slot_idx)
        self._free_slots.append(slot_idx)
        # self._debug_print(
        #     f"release slot={slot_idx} in_use={len(self._in_use_slots)} "
        #     f"high_water={self.high_water} slots={len(self._slots)} "
        #     f"free={len(self._free_slots)}"
        # )

    def iter_tensors(self) -> Iterator[torch.Tensor]:
        for slot in self._slots:
            for tensor in slot.values():
                yield tensor


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
    block: "OLMoDDPTransformerBlock"
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
        if _symm_mem is None and not olmo_symm_mem.is_enabled():
            raise RuntimeError("EP no-sync requires torch.distributed._symmetric_memory")

        slot_cache = self._slot_caches[slot_idx]
        cached = slot_cache.get(name)
        needs_realloc = cached is None or not _cached_symm_tensor_covers(cached, shape, dtype, device)
        # cache_state = "alloc" if needs_realloc else "cached"
        # _tbo_buffer_debug_print(
        #     f"shared_slot:{cache_state}:enter slot={slot_idx} name={name} "
        #     f"shape={shape} dtype={dtype} device={device}"
        # )
        if needs_realloc:
            symm_tensor = _alloc_ep_symm_tensor(
                shape=shape,
                dtype=dtype,
                device=device,
                group=self.group,
            )
            slot_cache[name] = symm_tensor
        # _tbo_buffer_debug_print(
        #     f"shared_slot:{cache_state}:exit slot={slot_idx} name={name} "
        #     f"cached_shape={tuple(slot_cache[name].shape)} return_shape={shape}"
        # )
        return _view_cached_symm_tensor(slot_cache[name], shape)

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

    def get_rowwise_fp8_dispatch_out_slot(
        self,
        *,
        slot_idx: int,
        dispatch_out_cap: int,
        d_model: int,
        block_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if slot_idx < 0 or slot_idx >= self.num_slots:
            raise ValueError(
                f"slot_idx must be in [0, {self.num_slots - 1}] (got {slot_idx})"
            )
        if d_model % block_size != 0:
            raise RuntimeError(
                "Rowwise FP8 requires hidden dim divisible by block_size: "
                f"hidden={d_model} block_size={block_size}"
            )
        dispatch_out_q = self._get_or_init_slot_tensor(
            slot_idx=slot_idx,
            name="dispatch_out_rowwise_fp8_q",
            shape=(dispatch_out_cap, d_model),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        dispatch_out_scales = self._get_or_init_slot_tensor(
            slot_idx=slot_idx,
            name="dispatch_out_rowwise_fp8_scales",
            shape=(dispatch_out_cap, d_model // block_size),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        return dispatch_out_q, dispatch_out_scales

    def get_rowwise_fp8_combine_in_slot(
        self,
        *,
        slot_idx: int,
        combine_in_cap: int,
        d_model: int,
        block_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if slot_idx < 0 or slot_idx >= self.num_slots:
            raise ValueError(
                f"slot_idx must be in [0, {self.num_slots - 1}] (got {slot_idx})"
            )
        if d_model % block_size != 0:
            raise RuntimeError(
                "Rowwise FP8 requires hidden dim divisible by block_size: "
                f"hidden={d_model} block_size={block_size}"
            )
        combine_in_q = self._get_or_init_slot_tensor(
            slot_idx=slot_idx,
            name="combine_in_rowwise_fp8_q",
            shape=(combine_in_cap, d_model),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        combine_in_scales = self._get_or_init_slot_tensor(
            slot_idx=slot_idx,
            name="combine_in_rowwise_fp8_scales",
            shape=(combine_in_cap, d_model // block_size),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        return combine_in_q, combine_in_scales

    def iter_tensors(self):
        for slot_cache in self._slot_caches:
            for tensor in slot_cache.values():
                yield tensor


def get_ep_no_sync_group_name(block: "OLMoDDPTransformerBlock") -> str:
    if not block.ep.no_sync:
        raise RuntimeError("EP no-sync is not enabled for this block")
    if block._ep_symm_group_name is None:
        raise RuntimeError(
            f"EP no-sync group is not initialized (block={block.block_idx}, ep_enabled={block.ep_enabled})"
        )
    return block._ep_symm_group_name


def ep_no_sync_slot_for_lane(block: "OLMoDDPTransformerBlock", lane_id: int) -> int:
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
    block: "OLMoDDPTransformerBlock",
    *,
    name: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if _symm_mem is None and not olmo_symm_mem.is_enabled():
        raise RuntimeError("EP no-sync requires torch.distributed._symmetric_memory")
    if block.ep_pg is None:
        raise RuntimeError("EP process group is not initialized")

    cached = block._ep_no_sync_symm_cache.get(name)
    needs_realloc = cached is None or not _cached_symm_tensor_covers(cached, shape, dtype, device)
    # cache_state = "alloc" if needs_realloc else "cached"
    # _tbo_buffer_debug_print(
    #     f"block_cache:{cache_state}:enter block={block.block_idx} name={name} "
    #     f"shape={shape} dtype={dtype} device={device}"
    # )
    if needs_realloc:
        try:
            symm_tensor = _alloc_ep_symm_tensor(
                shape=shape,
                dtype=dtype,
                device=device,
                group=block.ep_pg,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to allocate/rendezvous symmetric tensor '{name}' with shape={shape}, "
                f"dtype={dtype}, device={device}, block={block.block_idx}, rank={get_rank(block.ep_pg)}: {e}"
            ) from e
        block._ep_no_sync_symm_cache[name] = symm_tensor

    # _tbo_buffer_debug_print(
    #     f"block_cache:{cache_state}:exit block={block.block_idx} name={name} "
    #     f"cached_shape={tuple(block._ep_no_sync_symm_cache[name].shape)} return_shape={shape}"
    # )
    return _view_cached_symm_tensor(block._ep_no_sync_symm_cache[name], shape)


def _get_or_init_ep_no_sync_lease_pool(
    block: "OLMoDDPTransformerBlock",
    *,
    name: str,
) -> _NoSyncSymmLeasePool:
    if block.ep_pg is None:
        raise RuntimeError("EP process group is not initialized")
    pools = getattr(block, "_ep_no_sync_symm_lease_pools", None)
    if pools is None:
        pools = {}
        block._ep_no_sync_symm_lease_pools = pools
    pool = pools.get(name)
    if pool is None:
        pool = _NoSyncSymmLeasePool(name=f"block{block.block_idx}:{name}", group=block.ep_pg)
        pools[name] = pool
    return pool


def _rowwise_lifetime_lease_prewarm_slots(default: int = 1) -> int:
    raw = (
        os.getenv("OLMO_MOE_ROWWISE_LIFETIME_LEASE_SLOTS")
        or os.getenv("OLMO_MOE_ROWWISE_DISPATCH_OUT_LEASE_SLOTS")
        or str(default)
    ).strip()
    if not raw:
        return default
    try:
        slots = int(raw)
    except ValueError as e:
        raise RuntimeError(
            "OLMO_MOE_ROWWISE_LIFETIME_LEASE_SLOTS must be an integer"
        ) from e
    if slots < 0:
        raise RuntimeError(
            "OLMO_MOE_ROWWISE_LIFETIME_LEASE_SLOTS must be >= 0 "
            f"(got {slots})"
        )
    return slots


def _bf16_dispatch_out_specs(
    *,
    dispatch_out_cap: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[_NoSyncSymmLeaseTensorSpec, ...]:
    return (
        _NoSyncSymmLeaseTensorSpec(
            name="dispatch_out",
            shape=(dispatch_out_cap, d_model),
            dtype=dtype,
            device=device,
        ),
    )


def _bf16_combine_out_specs(
    *,
    combine_out_cap: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[_NoSyncSymmLeaseTensorSpec, ...]:
    return (
        _NoSyncSymmLeaseTensorSpec(
            name="combine_out",
            shape=(combine_out_cap, d_model),
            dtype=dtype,
            device=device,
        ),
    )


def _bf16_combine_gather_specs(
    *,
    combine_gather_cap: int,
    combine_gather_top_k: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[_NoSyncSymmLeaseTensorSpec, ...]:
    if combine_gather_cap <= 0 or combine_gather_top_k <= 0:
        raise RuntimeError(
            "combine_gather_cap and combine_gather_top_k must be positive "
            "when leasing combine_gather"
        )
    return (
        _NoSyncSymmLeaseTensorSpec(
            name="combine_gather",
            shape=(combine_gather_cap, combine_gather_top_k, d_model),
            dtype=dtype,
            device=device,
        ),
    )


def _fp8_dispatch_out_specs(
    *,
    dispatch_out_cap: int,
    d_model: int,
    block_size: int,
    device: torch.device,
) -> Tuple[_NoSyncSymmLeaseTensorSpec, ...]:
    if d_model % block_size != 0:
        raise RuntimeError(
            "Rowwise FP8 requires hidden dim divisible by block_size: "
            f"hidden={d_model} block_size={block_size}"
        )
    return (
        _NoSyncSymmLeaseTensorSpec(
            name="dispatch_out_q",
            shape=(dispatch_out_cap, d_model),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        _NoSyncSymmLeaseTensorSpec(
            name="dispatch_out_scales",
            shape=(dispatch_out_cap, d_model // block_size),
            dtype=torch.float8_e8m0fnu,
            device=device,
        ),
    )


@torch.compiler.disable
def acquire_ep_no_sync_dispatch_out_lease(
    block: "OLMoDDPTransformerBlock",
    *,
    dispatch_out_cap: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
) -> _NoSyncSymmLease:
    pool = _get_or_init_ep_no_sync_lease_pool(block, name="dispatch_out")
    return pool.acquire(
        specs=_bf16_dispatch_out_specs(
            dispatch_out_cap=dispatch_out_cap,
            d_model=d_model,
            dtype=dtype,
            device=device,
        ),
        owner="rowwise_dispatch_out",
    )


@torch.compiler.disable
def acquire_ep_no_sync_fp8_dispatch_out_lease(
    block: "OLMoDDPTransformerBlock",
    *,
    dispatch_out_cap: int,
    d_model: int,
    block_size: int,
    device: torch.device,
) -> _NoSyncSymmLease:
    pool = _get_or_init_ep_no_sync_lease_pool(block, name="dispatch_out_rowwise_fp8")
    return pool.acquire(
        specs=_fp8_dispatch_out_specs(
            dispatch_out_cap=dispatch_out_cap,
            d_model=d_model,
            block_size=block_size,
            device=device,
        ),
        owner="rowwise_fp8_dispatch_out",
    )


@torch.compiler.disable
def acquire_ep_no_sync_combine_out_lease(
    block: "OLMoDDPTransformerBlock",
    *,
    combine_out_cap: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
) -> _NoSyncSymmLease:
    pool = _get_or_init_ep_no_sync_lease_pool(block, name="combine_out")
    return pool.acquire(
        specs=_bf16_combine_out_specs(
            combine_out_cap=combine_out_cap,
            d_model=d_model,
            dtype=dtype,
            device=device,
        ),
        owner="rowwise_combine_out",
    )


@torch.compiler.disable
def acquire_ep_no_sync_combine_gather_lease(
    block: "OLMoDDPTransformerBlock",
    *,
    combine_gather_cap: int,
    combine_gather_top_k: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
) -> _NoSyncSymmLease:
    pool = _get_or_init_ep_no_sync_lease_pool(block, name="combine_gather")
    return pool.acquire(
        specs=_bf16_combine_gather_specs(
            combine_gather_cap=combine_gather_cap,
            combine_gather_top_k=combine_gather_top_k,
            d_model=d_model,
            dtype=dtype,
            device=device,
        ),
        owner="rowwise_combine_gather",
    )


@torch.compiler.disable
def prewarm_ep_no_sync_rowwise_dispatch_out_leases(
    block: "OLMoDDPTransformerBlock",
    *,
    dispatch_out_cap: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
    use_rowwise_fp8: bool,
    block_size: int = 0,
    num_slots: Optional[int] = None,
) -> None:
    num_slots = (
        _rowwise_lifetime_lease_prewarm_slots()
        if num_slots is None
        else num_slots
    )
    if num_slots == 0:
        return
    if use_rowwise_fp8:
        pool = _get_or_init_ep_no_sync_lease_pool(block, name="dispatch_out_rowwise_fp8")
        pool.prewarm(
            num_slots=num_slots,
            specs=_fp8_dispatch_out_specs(
                dispatch_out_cap=dispatch_out_cap,
                d_model=d_model,
                block_size=block_size,
                device=device,
            ),
        )
    else:
        pool = _get_or_init_ep_no_sync_lease_pool(block, name="dispatch_out")
        pool.prewarm(
            num_slots=num_slots,
            specs=_bf16_dispatch_out_specs(
                dispatch_out_cap=dispatch_out_cap,
                d_model=d_model,
                dtype=dtype,
                device=device,
            ),
        )


@torch.compiler.disable
def prewarm_ep_no_sync_rowwise_lifetime_leases(
    block: "OLMoDDPTransformerBlock",
    *,
    dispatch_out_cap: int,
    combine_out_cap: int,
    combine_gather_cap: int,
    combine_gather_top_k: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
    use_rowwise_fp8: bool,
    block_size: int = 0,
    need_dispatch_out: bool = True,
    need_combine_out: bool = False,
    need_combine_gather: bool = False,
    num_slots: Optional[int] = None,
) -> None:
    num_slots = (
        _rowwise_lifetime_lease_prewarm_slots()
        if num_slots is None
        else num_slots
    )
    if num_slots == 0:
        return
    if need_dispatch_out:
        prewarm_ep_no_sync_rowwise_dispatch_out_leases(
            block,
            dispatch_out_cap=dispatch_out_cap,
            d_model=d_model,
            dtype=dtype,
            device=device,
            use_rowwise_fp8=use_rowwise_fp8,
            block_size=block_size,
            num_slots=num_slots,
        )
    if need_combine_out:
        pool = _get_or_init_ep_no_sync_lease_pool(block, name="combine_out")
        pool.prewarm(
            num_slots=num_slots,
            specs=_bf16_combine_out_specs(
                combine_out_cap=combine_out_cap,
                d_model=d_model,
                dtype=dtype,
                device=device,
            ),
        )
    if need_combine_gather:
        pool = _get_or_init_ep_no_sync_lease_pool(block, name="combine_gather")
        pool.prewarm(
            num_slots=num_slots,
            specs=_bf16_combine_gather_specs(
                combine_gather_cap=combine_gather_cap,
                combine_gather_top_k=combine_gather_top_k,
                d_model=d_model,
                dtype=dtype,
                device=device,
            ),
        )


@torch.compiler.disable
def get_ep_no_sync_rowwise_fp8_buffers(
    block: "OLMoDDPTransformerBlock",
    *,
    dispatch_out_cap: int,
    combine_in_cap: int,
    d_model: int,
    block_size: int,
    device: torch.device,
    lease_dispatch_out: bool = False,
    need_dispatch_out: bool = True,
) -> _NoSyncRowwiseFP8SymmBuffers:
    if d_model % block_size != 0:
        raise RuntimeError(
            "Rowwise FP8 requires hidden dim divisible by block_size: "
            f"hidden={d_model} block_size={block_size}"
        )

    scale_cols = d_model // block_size
    dispatch_out_lease: Optional[_NoSyncSymmLease]
    if not need_dispatch_out:
        dispatch_out_lease = None
        dispatch_out_q = torch.empty((0,), dtype=torch.float8_e4m3fn, device=device)
        dispatch_out_scales = torch.empty((0,), dtype=torch.float8_e8m0fnu, device=device)
    elif lease_dispatch_out:
        dispatch_out_lease = acquire_ep_no_sync_fp8_dispatch_out_lease(
            block,
            dispatch_out_cap=dispatch_out_cap,
            d_model=d_model,
            block_size=block_size,
            device=device,
        )
        dispatch_out_q = dispatch_out_lease.tensor("dispatch_out_q")
        dispatch_out_scales = dispatch_out_lease.tensor("dispatch_out_scales")
    elif block._ep_no_sync_shared_pool is not None and block.ep.share_dispatch_out:
        dispatch_out_lease = None
        dispatch_out_q, dispatch_out_scales = (
            block._ep_no_sync_shared_pool.get_rowwise_fp8_dispatch_out_slot(
                slot_idx=block._ep_no_sync_shared_slot,
                dispatch_out_cap=dispatch_out_cap,
                d_model=d_model,
                block_size=block_size,
                device=device,
            )
        )
    else:
        dispatch_out_lease = None
        dispatch_out_q = get_or_init_ep_no_sync_symm_tensor(
            block,
            name="dispatch_out_rowwise_fp8_q",
            shape=(dispatch_out_cap, d_model),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        dispatch_out_scales = get_or_init_ep_no_sync_symm_tensor(
            block,
            name="dispatch_out_rowwise_fp8_scales",
            shape=(dispatch_out_cap, scale_cols),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
    if block._ep_no_sync_shared_pool is not None:
        # combine_in is scratch: forward does not save its contents and backward
        # overwrites it before use. Share one slot across layers instead of
        # pinning a full capacity-sized FP8 pair in every block cache.
        combine_in_q, combine_in_scales = (
            block._ep_no_sync_shared_pool.get_rowwise_fp8_combine_in_slot(
                slot_idx=block._ep_no_sync_shared_slot,
                combine_in_cap=combine_in_cap,
                d_model=d_model,
                block_size=block_size,
                device=device,
            )
        )
    else:
        combine_in_q = get_or_init_ep_no_sync_symm_tensor(
            block,
            name="combine_in_rowwise_fp8_q",
            shape=(combine_in_cap, d_model),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        combine_in_scales = get_or_init_ep_no_sync_symm_tensor(
            block,
            name="combine_in_rowwise_fp8_scales",
            shape=(combine_in_cap, scale_cols),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
    buffers = _NoSyncRowwiseFP8SymmBuffers(
        dispatch_out_q=dispatch_out_q,
        dispatch_out_scales=dispatch_out_scales,
        combine_in_q=combine_in_q,
        combine_in_scales=combine_in_scales,
        dispatch_out_lease=dispatch_out_lease,
    )
    if dispatch_out_lease is None and block._ep_no_sync_shared_pool is None:
        cache = getattr(block, "_ep_no_sync_rowwise_fp8_static_buffer_cache", None)
        if cache is not None:
            cache[
                _ep_no_sync_fp8_buffers_cache_key(
                    dispatch_out_cap=dispatch_out_cap,
                    combine_in_cap=combine_in_cap,
                    d_model=d_model,
                    block_size=block_size,
                    device=device,
                    need_dispatch_out=need_dispatch_out,
                )
            ] = buffers
    return buffers


def _parse_bool_env(value: str, *, env_name: str) -> Optional[bool]:
    normalized = value.strip().lower()
    if normalized in ("auto", ""):
        return None
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise RuntimeError(
        f"{env_name} must be one of auto|0|1|true|false|yes|no|on|off, got {value!r}"
    )


def _rowwise_symm_auto_enabled(block: "OLMoDDPTransformerBlock") -> bool:
    local_world_size = 0
    try:
        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "0") or "0")
    except ValueError:
        local_world_size = 0
    local_cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    intra_node_limit = max(local_world_size, local_cuda_devices, 8)
    return block.ep_world_size > intra_node_limit


def _resolve_rowwise_symm_option(
    block: "OLMoDDPTransformerBlock",
    *,
    attr_name: str,
    env_name: str,
    auto_enabled: bool = True,
) -> bool:
    if attr_name.startswith("ep."):
        configured = getattr(block.ep, attr_name.split(".", 1)[1], None)
    else:
        configured = getattr(block, attr_name, None)
    if configured is not None:
        return bool(configured)
    env_value = os.getenv(env_name)
    if env_value is not None:
        parsed = _parse_bool_env(env_value, env_name=env_name)
        if parsed is not None:
            return parsed
    return _rowwise_symm_auto_enabled(block) if auto_enabled else False


def resolve_ep_no_sync_rowwise_symm_options(block: "OLMoDDPTransformerBlock") -> None:
    """Resolve rowwise symmetric-buffer policy before compiled forwards run."""
    if block.ep.rowwise_symm_dispatch_in is None:
        block.ep.rowwise_symm_dispatch_in = _resolve_rowwise_symm_option(
            block,
            attr_name="ep.rowwise_symm_dispatch_in",
            env_name="OLMO_MOE_ROWWISE_SYMM_DISPATCH_IN",
        )
    if block.ep.rowwise_symm_combine_out is None:
        block.ep.rowwise_symm_combine_out = _resolve_rowwise_symm_option(
            block,
            attr_name="ep.rowwise_symm_combine_out",
            env_name="OLMO_MOE_ROWWISE_SYMM_COMBINE_OUT",
            auto_enabled=False,
        )
    if block.ep.rowwise_symm_combine_gather is None:
        block.ep.rowwise_symm_combine_gather = _resolve_rowwise_symm_option(
            block,
            attr_name="ep.rowwise_symm_combine_gather",
            env_name="OLMO_MOE_ROWWISE_SYMM_COMBINE_GATHER",
        )


def _ep_no_sync_buffers_cache_key(
    *,
    dispatch_in_cap: int,
    dispatch_out_cap: int,
    combine_in_cap: int,
    combine_out_cap: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
    slot_idx: Optional[int],
    need_dispatch_in: bool,
    need_dispatch_meta: bool,
    need_dispatch_out: bool,
    need_combine_in: bool,
    need_combine_meta: bool,
    need_combine_out: bool,
    need_combine_gather: bool,
    combine_gather_cap: int,
    combine_gather_top_k: int,
) -> Tuple[object, ...]:
    return (
        int(dispatch_in_cap),
        int(dispatch_out_cap),
        int(combine_in_cap),
        int(combine_out_cap),
        int(d_model),
        dtype,
        device,
        slot_idx,
        bool(need_dispatch_in),
        bool(need_dispatch_meta),
        bool(need_dispatch_out),
        bool(need_combine_in),
        bool(need_combine_meta),
        bool(need_combine_out),
        bool(need_combine_gather),
        int(combine_gather_cap),
        int(combine_gather_top_k),
    )


def _ep_no_sync_buffers_cache_slot(
    block: "OLMoDDPTransformerBlock",
    slot_idx: Optional[int],
) -> Optional[int]:
    if slot_idx is not None:
        return slot_idx
    if getattr(block, "_ep_no_sync_shared_pool", None) is not None:
        return block._ep_no_sync_shared_slot
    return None


def _ep_no_sync_fp8_buffers_cache_key(
    *,
    dispatch_out_cap: int,
    combine_in_cap: int,
    d_model: int,
    block_size: int,
    device: torch.device,
    need_dispatch_out: bool,
) -> Tuple[object, ...]:
    return (
        int(dispatch_out_cap),
        int(combine_in_cap),
        int(d_model),
        int(block_size),
        device,
        bool(need_dispatch_out),
    )


def get_cached_ep_no_sync_buffers(
    block: "OLMoDDPTransformerBlock",
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
    need_combine_gather: bool = False,
    combine_gather_cap: int = 0,
    combine_gather_top_k: int = 0,
) -> Optional[_NoSyncSymmBuffers]:
    cache = getattr(block, "_ep_no_sync_static_buffer_cache", None)
    if cache is None:
        return None
    return cache.get(
        _ep_no_sync_buffers_cache_key(
            dispatch_in_cap=dispatch_in_cap,
            dispatch_out_cap=dispatch_out_cap,
            combine_in_cap=combine_in_cap,
            combine_out_cap=combine_out_cap,
            d_model=d_model,
            dtype=dtype,
            device=device,
            slot_idx=_ep_no_sync_buffers_cache_slot(block, slot_idx),
            need_dispatch_in=need_dispatch_in,
            need_dispatch_meta=need_dispatch_meta,
            need_dispatch_out=need_dispatch_out,
            need_combine_in=need_combine_in,
            need_combine_meta=need_combine_meta,
            need_combine_out=need_combine_out,
            need_combine_gather=need_combine_gather,
            combine_gather_cap=combine_gather_cap,
            combine_gather_top_k=combine_gather_top_k,
        )
    )


def get_cached_ep_no_sync_rowwise_fp8_buffers(
    block: "OLMoDDPTransformerBlock",
    *,
    dispatch_out_cap: int,
    combine_in_cap: int,
    d_model: int,
    block_size: int,
    device: torch.device,
    need_dispatch_out: bool = True,
) -> Optional[_NoSyncRowwiseFP8SymmBuffers]:
    if getattr(block, "_ep_no_sync_shared_pool", None) is not None:
        # Shared-pool combine_in tensors can be resized by another block sharing
        # the same slot. Re-fetch the current slot views instead of caching
        # per-block dataclasses that may hold older tensor references.
        return None
    cache = getattr(block, "_ep_no_sync_rowwise_fp8_static_buffer_cache", None)
    if cache is None:
        return None
    return cache.get(
        _ep_no_sync_fp8_buffers_cache_key(
            dispatch_out_cap=dispatch_out_cap,
            combine_in_cap=combine_in_cap,
            d_model=d_model,
            block_size=block_size,
            device=device,
            need_dispatch_out=need_dispatch_out,
        )
    )


def use_ep_no_sync_rowwise_symm_dispatch_in(block: "OLMoDDPTransformerBlock") -> bool:
    return _resolve_rowwise_symm_option(
        block,
        attr_name="ep.rowwise_symm_dispatch_in",
        env_name="OLMO_MOE_ROWWISE_SYMM_DISPATCH_IN",
    )


def use_ep_no_sync_rowwise_symm_combine_out(block: "OLMoDDPTransformerBlock") -> bool:
    return _resolve_rowwise_symm_option(
        block,
        attr_name="ep.rowwise_symm_combine_out",
        env_name="OLMO_MOE_ROWWISE_SYMM_COMBINE_OUT",
        auto_enabled=False,
    )


def use_ep_no_sync_rowwise_symm_combine_gather(block: "OLMoDDPTransformerBlock") -> bool:
    return _resolve_rowwise_symm_option(
        block,
        attr_name="ep.rowwise_symm_combine_gather",
        env_name="OLMO_MOE_ROWWISE_SYMM_COMBINE_GATHER",
    )


@torch.compiler.disable
def acquire_ep_no_sync_rowwise_lifetime_leases(
    block: "OLMoDDPTransformerBlock",
    *,
    dispatch_out_cap: int,
    combine_out_cap: int,
    combine_gather_cap: int,
    combine_gather_top_k: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
    need_dispatch_out: bool,
    need_combine_out: bool,
    need_combine_gather: bool,
) -> _NoSyncRowwiseLifetimeLeases:
    return _NoSyncRowwiseLifetimeLeases(
        dispatch_out_lease=(
            acquire_ep_no_sync_dispatch_out_lease(
                block,
                dispatch_out_cap=dispatch_out_cap,
                d_model=d_model,
                dtype=dtype,
                device=device,
            )
            if need_dispatch_out
            else None
        ),
        combine_out_lease=(
            acquire_ep_no_sync_combine_out_lease(
                block,
                combine_out_cap=combine_out_cap,
                d_model=d_model,
                dtype=dtype,
                device=device,
            )
            if need_combine_out
            else None
        ),
        combine_gather_lease=(
            acquire_ep_no_sync_combine_gather_lease(
                block,
                combine_gather_cap=combine_gather_cap,
                combine_gather_top_k=combine_gather_top_k,
                d_model=d_model,
                dtype=dtype,
                device=device,
            )
            if need_combine_gather
            else None
        ),
    )


@torch.compiler.disable
def get_ep_no_sync_buffers(
    block: "OLMoDDPTransformerBlock",
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
    need_combine_gather: bool = False,
    combine_gather_cap: int = 0,
    combine_gather_top_k: int = 0,
    lease_dispatch_out: bool = False,
    lease_combine_out: bool = False,
    lease_combine_gather: bool = False,
) -> _NoSyncSymmBuffers:
    assert block.routed_experts_router is not None

    ep_world_size = block.ep_world_size
    transient_slot: Optional[_NoSyncSymmTransientSlot] = None
    resolved_slot_idx = block._ep_no_sync_shared_slot if slot_idx is None else slot_idx
    name_suffix = f"_slot{resolved_slot_idx}" if slot_idx is not None else ""
    chunk_reorder_backend = resolve_ep_no_sync_chunk_reorder_backend()
    # _tbo_buffer_debug_print(
    #     f"get_buffers:enter block={block.block_idx} slot={resolved_slot_idx} "
    #     f"suffix={name_suffix or '<none>'} shared_pool={block._ep_no_sync_shared_pool is not None} "
    #     f"need_dispatch_in={need_dispatch_in} need_dispatch_meta={need_dispatch_meta} "
    #     f"need_dispatch_out={need_dispatch_out} need_combine_in={need_combine_in} "
    #     f"need_combine_meta={need_combine_meta} need_combine_out={need_combine_out} "
    #     f"need_combine_gather={need_combine_gather}"
    # )
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
            # _tbo_buffer_debug_print(
            #     f"get_buffers:shared_slot-enter block={block.block_idx} slot={resolved_slot_idx}"
            # )
            transient_slot = block._ep_no_sync_shared_pool.get_slot(
                slot_idx=resolved_slot_idx,
                dispatch_in_cap=dispatch_in_cap,
                dispatch_out_cap=dispatch_out_cap,
                combine_in_cap=combine_in_cap,
                combine_out_cap=combine_out_cap,
                need_dispatch_in=need_dispatch_in,
                need_dispatch_meta=need_dispatch_meta,
                include_dispatch_out=(
                    need_dispatch_out and block.ep.share_dispatch_out and not lease_dispatch_out
                ),
                need_combine_in=need_combine_in,
                need_combine_meta=need_combine_meta,
                include_combine_out=(
                    need_combine_out and block.ep.share_combine_out and not lease_combine_out
                ),
                d_model=d_model,
                dtype=dtype,
                device=device,
                ep_world_size=ep_world_size,
            )
            # _tbo_buffer_debug_print(
            #     f"get_buffers:shared_slot-exit block={block.block_idx} slot={resolved_slot_idx}"
            # )
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
        if transient_slot is not None and transient_slot.dispatch_tmp_rank_splits_offsets is not None:
            dispatch_tmp_rank_splits_offsets = transient_slot.dispatch_tmp_rank_splits_offsets.detach()
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
        if transient_slot is not None and transient_slot.combine_tmp_rank_splits_offsets is not None:
            combine_tmp_rank_splits_offsets = transient_slot.combine_tmp_rank_splits_offsets.detach()
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

    dispatch_out_lease: Optional[_NoSyncSymmLease] = None
    if need_dispatch_out:
        shared_dispatch_out = transient_slot.dispatch_out if transient_slot is not None else None
        if lease_dispatch_out:
            dispatch_out_lease = acquire_ep_no_sync_dispatch_out_lease(
                block,
                dispatch_out_cap=dispatch_out_cap,
                d_model=d_model,
                dtype=dtype,
                device=device,
            )
            dispatch_out = dispatch_out_lease.tensor("dispatch_out")
            dispatch_out_is_shared = True
        elif shared_dispatch_out is not None:
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
        combine_out_lease: Optional[_NoSyncSymmLease] = None
        if lease_combine_out:
            combine_out_lease = acquire_ep_no_sync_combine_out_lease(
                block,
                combine_out_cap=combine_out_cap,
                d_model=d_model,
                dtype=dtype,
                device=device,
            )
            combine_out = combine_out_lease.tensor("combine_out")
            combine_out_is_shared = True
        elif shared_combine_out is not None:
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
        combine_out_lease = None

    if need_combine_gather:
        combine_gather_lease: Optional[_NoSyncSymmLease] = None
        if lease_combine_gather:
            combine_gather_lease = acquire_ep_no_sync_combine_gather_lease(
                block,
                combine_gather_cap=combine_gather_cap,
                combine_gather_top_k=combine_gather_top_k,
                d_model=d_model,
                dtype=dtype,
                device=device,
            )
            combine_gather = combine_gather_lease.tensor("combine_gather")
        elif combine_gather_cap <= 0 or combine_gather_top_k <= 0:
            raise RuntimeError(
                "combine_gather_cap and combine_gather_top_k must be positive "
                "when need_combine_gather=True"
            )
        else:
            combine_gather = get_or_init_ep_no_sync_symm_tensor(
                block,
                name=f"combine_gather{name_suffix}",
                shape=(combine_gather_cap, combine_gather_top_k, d_model),
                dtype=dtype,
                device=device,
            )
    else:
        combine_gather = empty_data
        combine_gather_lease = None

    # _tbo_buffer_debug_print(f"get_buffers:exit block={block.block_idx} slot={resolved_slot_idx}")
    buffers = _NoSyncSymmBuffers(
        dispatch_in=dispatch_in,
        dispatch_in_rank_splits=dispatch_in_rank_splits,
        dispatch_out=dispatch_out,
        dispatch_out_is_shared=dispatch_out_is_shared,
        dispatch_rank_splits_offsets=dispatch_rank_splits_offsets,
        dispatch_tmp_rank_splits_offsets=dispatch_tmp_rank_splits_offsets,
        combine_in=combine_in,
        combine_in_rank_splits=combine_in_rank_splits,
        combine_out=combine_out,
        combine_gather=combine_gather,
        combine_out_is_shared=combine_out_is_shared,
        combine_rank_splits_offsets=combine_rank_splits_offsets,
        combine_tmp_rank_splits_offsets=combine_tmp_rank_splits_offsets,
        dispatch_out_lease=dispatch_out_lease,
        combine_out_lease=combine_out_lease,
        combine_gather_lease=combine_gather_lease,
    )
    if not lease_dispatch_out and not lease_combine_out and not lease_combine_gather:
        cache = getattr(block, "_ep_no_sync_static_buffer_cache", None)
        if cache is not None:
            cache[
                _ep_no_sync_buffers_cache_key(
                    dispatch_in_cap=dispatch_in_cap,
                    dispatch_out_cap=dispatch_out_cap,
                    combine_in_cap=combine_in_cap,
                    combine_out_cap=combine_out_cap,
                    d_model=d_model,
                    dtype=dtype,
                    device=device,
                    slot_idx=_ep_no_sync_buffers_cache_slot(block, slot_idx),
                    need_dispatch_in=need_dispatch_in,
                    need_dispatch_meta=need_dispatch_meta,
                    need_dispatch_out=need_dispatch_out,
                    need_combine_in=need_combine_in,
                    need_combine_meta=need_combine_meta,
                    need_combine_out=need_combine_out,
                    need_combine_gather=need_combine_gather,
                    combine_gather_cap=combine_gather_cap,
                    combine_gather_top_k=combine_gather_top_k,
                )
            ] = buffers
    return buffers


def iter_ep_no_sync_symm_tensors(block: "OLMoDDPTransformerBlock") -> Iterator[torch.Tensor]:
    for tensor in block._ep_no_sync_symm_cache.values():
        if isinstance(tensor, torch.Tensor):
            yield tensor
    for pool in getattr(block, "_ep_no_sync_symm_lease_pools", {}).values():
        if isinstance(pool, _NoSyncSymmLeasePool):
            yield from pool.iter_tensors()
    if block._ep_no_sync_shared_pool is not None:
        yield from block._ep_no_sync_shared_pool.iter_tensors()


def compute_ep_no_sync_rank_capacity(block: "OLMoDDPTransformerBlock", num_out_tokens: int) -> int:
    # `num_out_tokens` is the local routed-token count before EP dispatch.
    # Under balanced routing, the average received tokens per EP rank is this
    # same value (not divided by ep_world_size).
    return max(
        1,
        int(math.ceil(block.ep.capacity_factor * float(num_out_tokens))),
    )
