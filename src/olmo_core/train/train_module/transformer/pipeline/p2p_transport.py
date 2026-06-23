from __future__ import annotations

from contextlib import contextmanager
import os
import socket
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.distributed as dist
import nvtx

from olmo_core.kernels import nccl_rma_p2p

P2PKey = tuple[str, int, int, int]


def _debug(message: str) -> None:
    if os.environ.get("OLMO_NCCL_RMA_P2P_DEBUG") != "1":
        return
    rank = dist.get_rank() if dist.is_initialized() else "?"
    print(f"[rank {rank} nccl-rma-p2p] {message}", flush=True)


def rma_group_hostnames(group: dist.ProcessGroup) -> set[str]:
    local_hostname = socket.gethostname().split(".")[0]
    hostnames: list[Optional[str]] = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(hostnames, local_hostname, group=group)
    return {hostname for hostname in hostnames if hostname is not None}


def rma_group_spans_nodes(group: dist.ProcessGroup) -> bool:
    return len(rma_group_hostnames(group)) > 1


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


def _send_label(key: P2PKey) -> str:
    kind, src_stage, _dst_stage, mb_index = key
    return f"{src_stage}{kind}{mb_index}-S"


def _recv_label(key: P2PKey) -> str:
    kind, _src_stage, dst_stage, mb_index = key
    return f"{dst_stage}{kind}{mb_index}-R"


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

    def _wait_event(self) -> None:
        if self._event is None:
            return
        # Keep the dependency on the CUDA stream instead of stalling the host
        # thread with cudaEventSynchronize.
        torch.cuda.current_stream().wait_event(self._event)

    def wait(self) -> None:
        raise NotImplementedError


class _RMASendWork(_RMAWork):
    def __init__(self, transport: "NCCLRMAPipelineP2PTransport", op: "RMASendOp") -> None:
        super().__init__()
        label = _send_label(op.key)
        _debug(f"send start key={op.key} peer={op.peer} offset={op.offset_bytes}")
        with transport.send_stream_context(wait_for_compute=True):
            op.tensor.record_stream(torch.cuda.current_stream())
            with nvtx.annotate(label, color="blue"):
                if op.wait_for_ack:
                    if op.ack_context_id is None:
                        raise RuntimeError("RMA send op requires an ack context but none was provided")
                    _debug(f"send wait ack key={op.key} peer={op.peer} channel={op.channel_index}")
                    with nvtx.annotate(f"{label}-wait-ack", color="purple"):
                        nccl_rma_p2p.wait_signal(op.ack_context_id, peer=op.peer, op_count=1)
                with nvtx.annotate(f"{label}-copy-in", color="purple"):
                    op.send_slot.copy_(op.tensor, non_blocking=True)
                with nvtx.annotate(f"{label}-put-signal", color="purple"):
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
        self._wait_event()


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
            label = _recv_label(self.op.key)
            _debug(
                "wait start "
                f"key={self.op.key} peer={self.op.peer} "
                f"op_count={self.op.expected_signal_count}"
            )
            self.op.output_tensor.record_stream(torch.cuda.current_stream())
            with nvtx.annotate(label, color="blue"):
                with nvtx.annotate(f"{label}-wait-signal", color="purple"):
                    nccl_rma_p2p.wait_signal(
                        self.transport.context_id,
                        peer=self.op.peer,
                        op_count=self.op.expected_signal_count,
                    )
                with nvtx.annotate(f"{label}-copy-out", color="purple"):
                    self.op.output_tensor.copy_(self.op.recv_slot, non_blocking=True)
                if self.op.ack_context_id is not None:
                    _debug(
                        "send ack "
                        f"key={self.op.key} peer={self.op.peer} channel={self.op.channel_index}"
                    )
                    with nvtx.annotate(f"{label}-ack", color="purple"):
                        nccl_rma_p2p.signal(self.op.ack_context_id, peer=self.op.peer)
                self._record_event()
            self._wait_enqueued = True
            _debug(
                "wait enqueued "
                f"key={self.op.key} peer={self.op.peer} "
                f"op_count={self.op.expected_signal_count}"
            )
        assert self._event is not None
        self._wait_event()
        _debug(
            f"wait dependency enqueued key={self.op.key} peer={self.op.peer} "
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
    channel_index: int
    wait_for_ack: bool
    ack_context_id: Optional[int]

    def start(self) -> _RMASendWork:
        return _RMASendWork(self.transport, self)


@dataclass
class RMARecvOp:
    transport: "NCCLRMAPipelineP2PTransport"
    key: P2PKey
    peer: int
    recv_slot: torch.Tensor
    output_tensor: torch.Tensor
    expected_signal_count: int
    channel_index: int
    ack_context_id: Optional[int]

    def start(self) -> _RMARecvWork:
        return _RMARecvWork(self.transport, self)


class NCCLRMAPipelineP2PTransport:
    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        device: torch.device,
        num_stages: int,
        use_ack: bool = False,
    ) -> None:
        if device.type != "cuda":
            raise RuntimeError("NCCL RMA P2P requires a CUDA device")

        self.group = group
        self.device = device
        self.group_rank = dist.get_rank(group)
        self.group_size = dist.get_world_size(group)
        self.num_stages = num_stages
        self.use_ack = use_ack
        self.hostnames = rma_group_hostnames(group)
        if len(self.hostnames) > 1:
            raise RuntimeError(
                "NCCL RMA P2P host APIs are not supported for this inter-node "
                f"communicator; group spans hosts {sorted(self.hostnames)}. "
                "Use p2p_backend=\"nccl\" for inter-node PP, or use the RMA "
                "backend only for single-node PP groups."
            )

        unique_id = _broadcast_nccl_unique_id(group, device)
        self.context_id = nccl_rma_p2p.init(
            unique_id,
            rank=self.group_rank,
            world_size=self.group_size,
            device=device.index if device.index is not None else torch.cuda.current_device(),
        )
        self.num_channels = 2 * max(num_stages - 1, 1)
        self.ack_context_ids: list[int] = []
        if self.use_ack:
            for channel_index in range(self.num_channels):
                ack_unique_id = _broadcast_nccl_unique_id(group, device)
                ack_context_id = nccl_rma_p2p.init(
                    ack_unique_id,
                    rank=self.group_rank,
                    world_size=self.group_size,
                    device=device.index if device.index is not None else torch.cuda.current_device(),
                )
                self.ack_context_ids.append(ack_context_id)
                _debug(f"initialized ack context channel={channel_index}")

        self.window_id: Optional[int] = None
        self.window: Optional[torch.Tensor] = None
        self.payload_shape: Optional[tuple[int, ...]] = None
        self.payload_dtype: Optional[torch.dtype] = None
        self.payload_numel = 0
        self.payload_nbytes = 0
        self.num_microbatches = 0
        self.slot_depth = 0
        self.num_slots = 0
        self._send_channel_started = [False for _ in range(self.num_channels)]
        with torch.cuda.device(device):
            self._send_stream = torch.cuda.Stream(priority=0)
        _debug(
            "initialized "
            f"group_rank={self.group_rank} group_size={self.group_size} "
            f"num_stages={self.num_stages} use_ack={self.use_ack} "
            "send_stream=dedicated recv_stream=compute"
        )

    @contextmanager
    def _stream_context(
        self,
        stream: torch.cuda.Stream,
        *,
        wait_for_compute: bool,
    ) -> Iterator[None]:
        with torch.cuda.device(self.device):
            if wait_for_compute:
                stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                yield

    def send_stream_context(self, *, wait_for_compute: bool) -> Iterator[None]:
        return self._stream_context(self._send_stream, wait_for_compute=wait_for_compute)

    def _synchronize_before_teardown(self) -> None:
        torch.cuda.synchronize(self.device)

    def close(self) -> None:
        self._synchronize_before_teardown()
        if self.window_id is not None:
            nccl_rma_p2p.free_window(self.context_id, self.window_id)
            self.window_id = None
            self.window = None
        for ack_context_id in self.ack_context_ids:
            nccl_rma_p2p.destroy(ack_context_id)
        self.ack_context_ids = []
        nccl_rma_p2p.destroy(self.context_id)

    def prepare_step(
        self,
        *,
        num_microbatches: int,
        payload_shape: tuple[int, ...],
        payload_dtype: torch.dtype,
        slot_depth: int = 2,
    ) -> None:
        if num_microbatches < 1:
            raise RuntimeError(f"NCCL RMA P2P requires at least one microbatch, got {num_microbatches}")
        if slot_depth < 1:
            raise RuntimeError(f"NCCL RMA P2P slot_depth must be >= 1, got {slot_depth}")
        if self.use_ack and slot_depth != 1:
            raise RuntimeError("NCCL RMA ack mode requires slot_depth=1")
        slot_depth = min(slot_depth, num_microbatches)

        if (
            self.window is not None
            and self.slot_depth == slot_depth
            and self.payload_shape == payload_shape
            and self.payload_dtype == payload_dtype
        ):
            self.num_microbatches = num_microbatches
            return

        if self.window_id is not None:
            self._synchronize_before_teardown()
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
        self.slot_depth = slot_depth
        self.num_slots = self.num_channels * slot_depth

        window_id, window = nccl_rma_p2p.alloc_window(
            self.context_id,
            (self.num_slots, *payload_shape),
            dtype=_dtype_name(payload_dtype),
        )
        self.window_id = window_id
        self.window = window
        _debug(
            "prepared window "
            f"num_slots={self.num_slots} slot_depth={slot_depth} payload_shape={payload_shape} "
            f"payload_dtype={payload_dtype}"
        )

    def _channel_index(self, key: P2PKey) -> int:
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

        return direction * (self.num_stages - 1) + edge_index

    def _slot_index(self, key: P2PKey) -> int:
        channel_index = self._channel_index(key)
        _kind, _src_stage, _dst_stage, mb_index = key
        slot_lane = mb_index % self.slot_depth
        return channel_index * self.slot_depth + slot_lane

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
        channel_index = self._channel_index(key)
        slot_index, send_slot = self._slot(key)
        wait_for_ack = False
        ack_context_id: Optional[int] = None
        if self.use_ack:
            wait_for_ack = self._send_channel_started[channel_index]
            self._send_channel_started[channel_index] = True
            ack_context_id = self.ack_context_ids[channel_index]
        _debug(f"make send key={key} peer={peer} slot={slot_index}")
        return RMASendOp(
            transport=self,
            key=key,
            peer=peer,
            tensor=tensor.contiguous(),
            send_slot=send_slot,
            offset_bytes=slot_index * self.payload_nbytes,
            channel_index=channel_index,
            wait_for_ack=wait_for_ack,
            ack_context_id=ack_context_id,
        )

    def make_recv_op(
        self,
        key: P2PKey,
        *,
        peer: int,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> RMARecvOp:
        channel_index = self._channel_index(key)
        slot_index, recv_slot = self._slot(key)
        if self.payload_shape is None or self.payload_dtype is None:
            raise RuntimeError("NCCL RMA P2P transport has not been prepared for this step")
        if output_tensor is None:
            output_tensor = torch.empty(
                self.payload_shape,
                device=self.device,
                dtype=self.payload_dtype,
            )
        if not output_tensor.is_cuda:
            raise RuntimeError("NCCL RMA P2P receive output tensor must be CUDA")
        if tuple(output_tensor.size()) != self.payload_shape:
            raise RuntimeError(
                f"NCCL RMA P2P receive output shape {tuple(output_tensor.size())} "
                f"does not match {self.payload_shape}"
            )
        if output_tensor.dtype != self.payload_dtype:
            raise RuntimeError(
                f"NCCL RMA P2P receive output dtype {output_tensor.dtype} "
                f"does not match {self.payload_dtype}"
            )
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
            output_tensor=output_tensor,
            expected_signal_count=1,
            channel_index=channel_index,
            ack_context_id=self.ack_context_ids[channel_index] if self.use_ack else None,
        )
