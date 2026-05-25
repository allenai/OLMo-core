from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from olmo_core.kernels import nccl_rma_p2p

P2PKey = tuple[str, int, int, int]


def _debug(message: str) -> None:
    if os.environ.get("OLMO_NCCL_RMA_P2P_DEBUG") != "1":
        return
    rank = dist.get_rank() if dist.is_initialized() else "?"
    print(f"[rank {rank} nccl-rma-p2p] {message}", flush=True)


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.int32:
        return "int32"
    if dtype == torch.int64:
        return "int64"
    if dtype == torch.uint8:
        return "uint8"
    raise RuntimeError(f"Unsupported NCCL RMA P2P dtype: {dtype}")


def _broadcast_nccl_unique_id(group: dist.ProcessGroup, device: torch.device) -> bytes:
    rank = dist.get_rank(group)
    src_global_rank = dist.get_global_rank(group, 0)
    if rank == 0:
        unique_id = nccl_rma_p2p.get_unique_id()
        unique_id_tensor = torch.tensor(
            list(unique_id),
            dtype=torch.uint8,
            device=device,
        )
    else:
        unique_id_tensor = torch.empty(128, dtype=torch.uint8, device=device)

    dist.broadcast(unique_id_tensor, src=src_global_rank, group=group)
    return bytes(unique_id_tensor.cpu().tolist())


class _RMAWork:
    def __init__(self) -> None:
        self._event: Optional[torch.cuda.Event] = None

    def _record_event(self) -> None:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self._event = event

    def is_completed(self) -> bool:
        return self._event is not None and self._event.query()

    def wait(self) -> None:
        raise NotImplementedError


class _RMASendWork(_RMAWork):
    def __init__(self, transport: "NCCLRMAPipelineP2PTransport", op: "RMASendOp") -> None:
        super().__init__()
        _debug(f"send start key={op.key} peer={op.peer} offset={op.offset_bytes}")
        op.send_slot.copy_(op.tensor, non_blocking=True)
        nccl_rma_p2p.put_signal(
            transport.context_id,
            op.send_slot,
            peer=op.peer,
            window_id=transport.window_id,
            peer_window_offset_bytes=op.offset_bytes,
        )
        self._record_event()
        _debug(f"send enqueued key={op.key} peer={op.peer} offset={op.offset_bytes}")

    def wait(self) -> None:
        if self._event is not None:
            self._event.synchronize()


class _RMARecvWork(_RMAWork):
    def __init__(self, transport: "NCCLRMAPipelineP2PTransport", op: "RMARecvOp") -> None:
        super().__init__()
        self.transport = transport
        self.op = op
        self._wait_enqueued = False

    def is_completed(self) -> bool:
        return self._wait_enqueued and super().is_completed()

    def wait(self) -> None:
        if not self._wait_enqueued:
            _debug(
                "wait start "
                f"key={self.op.key} peer={self.op.peer} "
                f"op_count={self.op.expected_signal_count}"
            )
            nccl_rma_p2p.wait_signal(
                self.transport.context_id,
                peer=self.op.peer,
                op_count=self.op.expected_signal_count,
            )
            self._record_event()
            self._wait_enqueued = True
            _debug(
                "wait enqueued "
                f"key={self.op.key} peer={self.op.peer} "
                f"op_count={self.op.expected_signal_count}"
            )
        assert self._event is not None
        self._event.synchronize()
        _debug(
            "wait done "
            f"key={self.op.key} peer={self.op.peer} "
            f"op_count={self.op.expected_signal_count}"
        )


@dataclass
class RMASendOp:
    transport: "NCCLRMAPipelineP2PTransport"
    key: P2PKey
    peer: int
    tensor: torch.Tensor
    send_slot: torch.Tensor
    offset_bytes: int

    def start(self) -> _RMASendWork:
        return _RMASendWork(self.transport, self)


@dataclass
class RMARecvOp:
    transport: "NCCLRMAPipelineP2PTransport"
    key: P2PKey
    peer: int
    recv_slot: torch.Tensor
    expected_signal_count: int

    def start(self) -> _RMARecvWork:
        return _RMARecvWork(self.transport, self)


class NCCLRMAPipelineP2PTransport:
    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        device: torch.device,
        num_stages: int,
    ) -> None:
        if device.type != "cuda":
            raise RuntimeError("NCCL RMA P2P requires a CUDA device")

        self.group = group
        self.device = device
        self.group_rank = dist.get_rank(group)
        self.group_size = dist.get_world_size(group)
        self.num_stages = num_stages

        unique_id = _broadcast_nccl_unique_id(group, device)
        self.context_id = nccl_rma_p2p.init(
            unique_id,
            rank=self.group_rank,
            world_size=self.group_size,
            device=device.index if device.index is not None else torch.cuda.current_device(),
        )

        self.window_id: Optional[int] = None
        self.window: Optional[torch.Tensor] = None
        self.payload_shape: Optional[tuple[int, ...]] = None
        self.payload_dtype: Optional[torch.dtype] = None
        self.payload_numel = 0
        self.payload_nbytes = 0
        self.num_microbatches = 0
        self.num_slots = 0
        _debug(
            "initialized "
            f"group_rank={self.group_rank} group_size={self.group_size} "
            f"num_stages={self.num_stages}"
        )

    def close(self) -> None:
        if self.window_id is not None:
            nccl_rma_p2p.free_window(self.context_id, self.window_id)
            self.window_id = None
            self.window = None
        nccl_rma_p2p.destroy(self.context_id)

    def prepare_step(
        self,
        *,
        num_microbatches: int,
        payload_shape: tuple[int, ...],
        payload_dtype: torch.dtype,
    ) -> None:
        if (
            self.window is not None
            and self.num_microbatches == num_microbatches
            and self.payload_shape == payload_shape
            and self.payload_dtype == payload_dtype
        ):
            return

        if self.window_id is not None:
            nccl_rma_p2p.free_window(self.context_id, self.window_id)

        self.num_microbatches = num_microbatches
        self.payload_shape = payload_shape
        self.payload_dtype = payload_dtype
        self.payload_numel = 1
        for dim in payload_shape:
            self.payload_numel *= dim
        self.payload_nbytes = (
            self.payload_numel * torch.empty((), dtype=payload_dtype).element_size()
        )
        self.num_slots = 2 * max(self.num_stages - 1, 1) * num_microbatches

        window_id, window = nccl_rma_p2p.alloc_window(
            self.context_id,
            (self.num_slots, *payload_shape),
            dtype=_dtype_name(payload_dtype),
        )
        self.window_id = window_id
        self.window = window
        _debug(
            "prepared window "
            f"num_slots={self.num_slots} payload_shape={payload_shape} "
            f"payload_dtype={payload_dtype}"
        )

    def _slot_index(self, key: P2PKey) -> int:
        kind, src_stage, dst_stage, mb_index = key
        if kind == "F":
            direction = 0
        elif kind == "B":
            direction = 1
        else:
            raise RuntimeError(f"Unexpected P2P key kind: {kind}")

        edge_index = min(src_stage, dst_stage)
        if not (0 <= edge_index < self.num_stages - 1):
            raise RuntimeError(f"Invalid P2P edge in key: {key}")
        if not (0 <= mb_index < self.num_microbatches):
            raise RuntimeError(f"Invalid P2P microbatch in key: {key}")

        return (direction * (self.num_stages - 1) + edge_index) * self.num_microbatches + mb_index

    def _slot(self, key: P2PKey) -> tuple[int, torch.Tensor]:
        if self.window is None:
            raise RuntimeError("NCCL RMA P2P transport has not been prepared for this step")
        slot_index = self._slot_index(key)
        return slot_index, self.window[slot_index]

    def make_send_op(self, key: P2PKey, *, peer: int, tensor: torch.Tensor) -> RMASendOp:
        if self.window_id is None:
            raise RuntimeError("NCCL RMA P2P transport has no registered window")
        if not tensor.is_cuda:
            raise RuntimeError("NCCL RMA P2P send tensor must be CUDA")
        if tuple(tensor.size()) != self.payload_shape:
            raise RuntimeError(
                f"NCCL RMA P2P tensor shape {tuple(tensor.size())} does not match {self.payload_shape}"
            )
        if tensor.dtype != self.payload_dtype:
            raise RuntimeError(f"NCCL RMA P2P tensor dtype {tensor.dtype} does not match {self.payload_dtype}")
        slot_index, send_slot = self._slot(key)
        _debug(f"make send key={key} peer={peer} slot={slot_index}")
        return RMASendOp(
            transport=self,
            key=key,
            peer=peer,
            tensor=tensor.contiguous(),
            send_slot=send_slot,
            offset_bytes=slot_index * self.payload_nbytes,
        )

    def make_recv_op(self, key: P2PKey, *, peer: int) -> RMARecvOp:
        slot_index, recv_slot = self._slot(key)
        # NCCL 2.29.7 exposes only signal index/context 0, so waits consume one
        # signal from the peer-wide signal queue. Correctness depends on keeping
        # per-peer send order and wait order consistent in the pipeline schedule.
        _debug(
            "make recv "
            f"key={key} peer={peer} slot={slot_index} "
            "op_count=1"
        )
        return RMARecvOp(
            transport=self,
            key=key,
            peer=peer,
            recv_slot=recv_slot,
            expected_signal_count=1,
        )
