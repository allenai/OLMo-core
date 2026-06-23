from __future__ import annotations

from collections import defaultdict
import logging
import os
from typing import Any, Iterable, Optional

import nvtx
import torch
import torch.distributed as dist

from .p2p_transport import NCCLRMAPipelineP2PTransport, P2PKey, rma_group_hostnames

logger = logging.getLogger(__name__)

KeyedP2POp = tuple[P2PKey, str, Any]
PendingP2PEntry = tuple[Any, Any, str, int]


def _op_label(key: P2PKey, op_kind: str) -> str:
    kind, src_stage, dst_stage, mb_index = key
    if op_kind.endswith("SEND"):
        return f"{src_stage}{kind}{mb_index}-S"
    return f"{dst_stage}{kind}{mb_index}-R"


def _launch_label(keyed_ops: Iterable[KeyedP2POp]) -> str:
    labels: list[str] = []
    seen: set[str] = set()
    for key, op_kind, _op in keyed_ops:
        label = _op_label(key, op_kind)
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return ",".join(labels)


class PipelineP2PExecutor:
    should_prefetch_next_action_inputs = False
    transport: Optional[NCCLRMAPipelineP2PTransport] = None

    def __init__(
        self,
        *,
        backend: str,
        rank: int,
    ) -> None:
        self.backend = backend
        self.rank = rank
        self.pending: dict[P2PKey, list[PendingP2PEntry]] = defaultdict(list)

    @property
    def has_pending(self) -> bool:
        return bool(self.pending)

    def prepare_step(
        self,
        *,
        stages: list[Any],
        num_microbatches: int,
        slot_depth: int,
    ) -> None:
        del stages, num_microbatches, slot_depth

    def debug(self, message: str) -> None:
        del message

    def wait_all(self) -> None:
        for key in list(self.pending):
            self.wait_key(key)

    def wait_key(self, key: P2PKey) -> None:
        self.debug(f"wait key={key} entries={len(self.pending.get(key, []))}")
        for handle, _op, _op_kind, _launch_overlap_step in self.pending.pop(key, []):
            handle.wait()

    def maybe_wait_key(self, key: P2PKey) -> bool:
        if key not in self.pending:
            return False
        self.wait_key(key)
        return True

    def wait_ops(self, op_kinds: set[str] | frozenset[str]) -> None:
        for key in list(self.pending):
            pending = self.pending.pop(key)
            still_pending: list[PendingP2PEntry] = []
            for handle, op, op_kind, launch_overlap_step in pending:
                if op_kind in op_kinds:
                    handle.wait()
                else:
                    still_pending.append((handle, op, op_kind, launch_overlap_step))
            if still_pending:
                self.pending[key] = still_pending

    def prune_completed(self) -> None:
        for key, entries in list(self.pending.items()):
            pending: list[PendingP2PEntry] = []
            for handle, op, op_kind, launch_overlap_step in entries:
                if handle.is_completed():
                    handle.wait()
                else:
                    pending.append((handle, op, op_kind, launch_overlap_step))
            if pending:
                self.pending[key] = pending
            else:
                self.pending.pop(key, None)

    def wait_launched_at_or_before(self, overlap_compute_step: int) -> None:
        for key in list(self.pending):
            pending = self.pending.pop(key)
            still_pending: list[PendingP2PEntry] = []
            for handle, op, op_kind, launch_overlap_step in pending:
                force_wait = launch_overlap_step <= overlap_compute_step
                if self._should_force_wait(op_kind, force_wait):
                    handle.wait()
                else:
                    still_pending.append((handle, op, op_kind, launch_overlap_step))
            if still_pending:
                self.pending[key] = still_pending

    def launch(
        self,
        keyed_ops: list[KeyedP2POp],
        *,
        completed_p2p_overlap_compute_steps: int,
        debug_context: str = "",
    ) -> None:
        if not keyed_ops:
            return
        keyed_ops = self._order_ops_for_launch(keyed_ops)
        with nvtx.annotate(_launch_label(keyed_ops), color="blue"):
            self.debug(
                "launch "
                f"{debug_context} "
                f"kinds={[op_kind for _key, op_kind, _op in keyed_ops]} "
                f"keys={[key for key, _op_kind, _op in keyed_ops]}"
            )
            handles = self._start_ops(keyed_ops)

        if len(handles) == 1 and len(keyed_ops) > 1:
            handles_for_ops = handles * len(keyed_ops)
        elif len(handles) == len(keyed_ops):
            handles_for_ops = handles
        else:
            raise RuntimeError(
                "Unexpected number of P2P work handles: "
                f"got {len(handles)} handles for {len(keyed_ops)} ops"
            )

        for (key, op_kind, op), handle in zip(keyed_ops, handles_for_ops):
            self.pending[key].append(
                (
                    handle,
                    op,
                    op_kind,
                    completed_p2p_overlap_compute_steps,
                )
            )

    def _start_ops(self, keyed_ops: list[KeyedP2POp]) -> list[Any]:
        return dist.batch_isend_irecv([op for _key, _op_kind, op in keyed_ops])

    def _order_ops_for_launch(self, keyed_ops: list[KeyedP2POp]) -> list[KeyedP2POp]:
        return keyed_ops

    def _should_force_wait(self, _op_kind: str, force_wait: bool) -> bool:
        return force_wait


class TorchDistP2PExecutor(PipelineP2PExecutor):
    def __init__(self, *, rank: int) -> None:
        super().__init__(backend="nccl", rank=rank)


class NCCLRMAP2PExecutor(PipelineP2PExecutor):
    should_prefetch_next_action_inputs = True

    def __init__(
        self,
        *,
        backend: str,
        rank: int,
        stages: list[Any],
        use_ack: bool,
    ) -> None:
        super().__init__(backend=backend, rank=rank)
        self.transport = NCCLRMAPipelineP2PTransport(
            group=stages[0].p2p_group,
            device=stages[0].device,
            num_stages=stages[0].num_stages,
            use_ack=use_ack,
        )
        for stage in stages:
            stage.set_p2p_transport(self.transport)

    def prepare_step(
        self,
        *,
        stages: list[Any],
        num_microbatches: int,
        slot_depth: int,
    ) -> None:
        payload_meta: Optional[torch.Tensor] = None
        for stage in stages:
            for meta in (stage.outputs_meta, stage.inputs_meta):
                if meta is not None and meta.dtype == stage.p2p_dtype and meta.dim() == 3:
                    payload_meta = meta
                    break
            if payload_meta is not None:
                break
        if payload_meta is None:
            raise RuntimeError("Could not infer NCCL RMA P2P payload metadata")
        assert self.transport is not None
        self.transport.prepare_step(
            num_microbatches=num_microbatches,
            payload_shape=tuple(payload_meta.size()),
            payload_dtype=payload_meta.dtype,
            slot_depth=slot_depth,
        )

    def debug(self, message: str) -> None:
        if os.environ.get("OLMO_NCCL_RMA_P2P_DEBUG") != "1":
            return
        print(f"[rank {self.rank} pipeline-rma] {message}", flush=True)

    def _start_ops(self, keyed_ops: list[KeyedP2POp]) -> list[Any]:
        return [op.start() for _key, _op_kind, op in keyed_ops]

    def _order_ops_for_launch(self, keyed_ops: list[KeyedP2POp]) -> list[KeyedP2POp]:
        recvs: list[KeyedP2POp] = []
        sends: list[KeyedP2POp] = []
        other: list[KeyedP2POp] = []
        for keyed_op in keyed_ops:
            _key, op_kind, _op = keyed_op
            if op_kind.endswith("RECV"):
                recvs.append(keyed_op)
            elif op_kind.endswith("SEND"):
                sends.append(keyed_op)
            else:
                other.append(keyed_op)
        return recvs + sends + other

    def _should_force_wait(self, op_kind: str, force_wait: bool) -> bool:
        if force_wait and op_kind.endswith("RECV"):
            # RMA receive copy-out/ack runs on the compute stream just before
            # the consumer action. The overlap budget should not pull later
            # receives ahead of unrelated local compute.
            return False
        return force_wait


def build_p2p_executor(
    *,
    backend: str,
    rank: int,
    stages: list[Any],
) -> PipelineP2PExecutor:
    if backend == "nccl":
        return TorchDistP2PExecutor(rank=rank)
    if backend in {"nccl_rma", "nccl_rma_ack"}:
        hostnames = rma_group_hostnames(stages[0].p2p_group)
        if len(hostnames) > 1:
            logger.warning(
                "p2p_backend=%r requested, but NCCL host-enqueued RMA is not "
                "supported for inter-node communicators in this environment "
                "(hosts=%s). Falling back to p2p_backend=\"nccl\".",
                backend,
                sorted(hostnames),
            )
            for stage in stages:
                stage.p2p_backend = "nccl"
            return TorchDistP2PExecutor(rank=rank)
        return NCCLRMAP2PExecutor(
            backend=backend,
            rank=rank,
            stages=stages,
            use_ack=backend == "nccl_rma_ack",
        )
    raise RuntimeError(f"Unsupported custom pipeline P2P backend: {backend}")
