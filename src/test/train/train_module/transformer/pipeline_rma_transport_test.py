from __future__ import annotations

import torch
import pytest

from olmo_core.train.train_module.transformer.pipeline import p2p_transport


def _make_uninitialized_transport() -> p2p_transport.NCCLRMAPipelineP2PTransport:
    transport = p2p_transport.NCCLRMAPipelineP2PTransport.__new__(
        p2p_transport.NCCLRMAPipelineP2PTransport
    )
    transport.context_id = 0
    transport.window_id = None
    transport.window = None
    transport.device = torch.device("cpu")
    transport.num_stages = 4
    transport.use_ack = False
    transport.num_channels = 6
    transport.ack_context_ids = []
    transport.payload_shape = None
    transport.payload_dtype = None
    transport.payload_numel = 0
    transport.payload_nbytes = 0
    transport.num_microbatches = 0
    transport.slot_depth = 0
    transport.num_slots = 0
    transport._send_channel_started = [False for _ in range(transport.num_channels)]
    return transport


def test_rma_window_slots_are_bounded_by_slot_depth(monkeypatch):
    alloc_calls: list[tuple[int, ...]] = []

    def fake_alloc_window(context_id, shape, dtype):
        del context_id, dtype
        alloc_calls.append(shape)
        return 7, torch.empty(shape)

    monkeypatch.setattr(p2p_transport.nccl_rma_p2p, "alloc_window", fake_alloc_window)

    transport = _make_uninitialized_transport()
    transport.prepare_step(
        num_microbatches=100,
        payload_shape=(2, 3),
        payload_dtype=torch.float32,
        slot_depth=2,
    )

    assert transport.num_slots == 12
    assert alloc_calls == [(12, 2, 3)]
    assert transport._slot_index(("F", 0, 1, 0)) == transport._slot_index(("F", 0, 1, 2))
    assert transport._slot_index(("F", 0, 1, 1)) != transport._slot_index(("F", 0, 1, 2))


def test_rma_ack_window_uses_one_lane(monkeypatch):
    alloc_calls: list[tuple[int, ...]] = []

    def fake_alloc_window(context_id, shape, dtype):
        del context_id, dtype
        alloc_calls.append(shape)
        return 7, torch.empty(shape)

    monkeypatch.setattr(p2p_transport.nccl_rma_p2p, "alloc_window", fake_alloc_window)

    transport = _make_uninitialized_transport()
    transport.use_ack = True
    transport.ack_context_ids = list(range(transport.num_channels))
    transport.prepare_step(
        num_microbatches=100,
        payload_shape=(2, 3),
        payload_dtype=torch.float32,
        slot_depth=1,
    )

    assert transport.num_slots == 6
    assert alloc_calls == [(6, 2, 3)]
    assert transport._slot_index(("F", 0, 1, 0)) == transport._slot_index(("F", 0, 1, 99))


def test_rma_ack_rejects_multi_lane_depth():
    transport = _make_uninitialized_transport()
    transport.use_ack = True
    with pytest.raises(RuntimeError, match="requires slot_depth=1"):
        transport.prepare_step(
            num_microbatches=100,
            payload_shape=(2, 3),
            payload_dtype=torch.float32,
            slot_depth=2,
        )
