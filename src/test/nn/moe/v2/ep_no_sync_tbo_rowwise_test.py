from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from olmo_core.nn.moe.v2 import ep_no_sync_tbo_rowwise as rowwise_tbo
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath


def test_rowwise_tbo_fails_closed_for_fp8():
    block = SimpleNamespace(
        ep=SimpleNamespace(path=ExpertParallelPath.rowwise_nvshmem),
        rowwise_fp8=SimpleNamespace(enabled=True),
    )

    with pytest.raises(NotImplementedError, match="Rowwise FP8"):
        rowwise_tbo._check_rowwise_tbo_supported(block)  # type: ignore[arg-type]


def test_rowwise_tbo_combined_forward_fresh_schedule(monkeypatch):
    calls: list = []
    fake_block = SimpleNamespace()

    def fake_stage_a(block, x, *, lane_id, loss_div_factor=None, **kwargs):
        calls.append(("a", lane_id, x, loss_div_factor, kwargs))
        return SimpleNamespace(lane_id=lane_id)

    def fake_stage_d(block, a_state):
        calls.append(("d", a_state.lane_id))
        return SimpleNamespace(lane_id=a_state.lane_id)

    def fake_stage_e(block, d_state):
        calls.append(("e", d_state.lane_id))
        return rowwise_tbo._NoSyncRowwiseTboPendingContext(
            block=block,
            lane_id=d_state.lane_id,
            a_state=SimpleNamespace(lane_id=d_state.lane_id),  # type: ignore[arg-type]
            global_x_rank_major=torch.tensor([[float(d_state.lane_id)]]),
        )

    def fake_stage_c(block, pending):
        calls.append(("c", pending.lane_id))
        pending.combine_out = torch.tensor([[10.0 + pending.lane_id]])
        pending.combine_done_event = object()
        return pending

    def fake_tail(block, pending):
        calls.append(("tail", pending.lane_id))
        return f"final:{pending.lane_id}"

    monkeypatch.setattr(rowwise_tbo, "ep_no_sync_rowwise_tbo_stage_a", fake_stage_a)
    monkeypatch.setattr(rowwise_tbo, "ep_no_sync_rowwise_tbo_stage_d_launch", fake_stage_d)
    monkeypatch.setattr(rowwise_tbo, "ep_no_sync_rowwise_tbo_stage_e", fake_stage_e)
    monkeypatch.setattr(rowwise_tbo, "ep_no_sync_rowwise_tbo_stage_c_launch", fake_stage_c)
    monkeypatch.setattr(rowwise_tbo, "ep_no_sync_rowwise_tbo_stage_tail", fake_tail)

    out, pending = rowwise_tbo.combined_forward_ep_no_sync_tbo_rowwise(
        fake_block,  # type: ignore[arg-type]
        "x0",
        {"x1": "x1"},
        True,
        loss_div_factor=7.0,
        attention_mask="mask",
    )

    assert out == "final:0"
    assert isinstance(pending, rowwise_tbo._NoSyncRowwiseTboPendingContext)
    assert pending.lane_id == 1
    assert calls == [
        ("a", 0, "x0", 7.0, {"attention_mask": "mask"}),
        ("d", 0),
        ("a", 1, "x1", 7.0, {"attention_mask": "mask"}),
        ("d", 1),
        ("e", 0),
        ("c", 0),
        ("e", 1),
        ("tail", 0),
    ]


def test_rowwise_stage_d_launch_uses_comm_stream_and_dispatch(monkeypatch):
    calls: list = []
    event = object()
    comm_stream = object()

    monkeypatch.setattr(rowwise_tbo, "get_or_init_stream", lambda **kwargs: comm_stream)
    monkeypatch.setattr(
        rowwise_tbo,
        "wait_stream_no_compile",
        lambda this_stream, other_stream: calls.append(("wait_stream", this_stream, other_stream)),
    )
    monkeypatch.setattr(
        rowwise_tbo.torch.cuda,
        "current_stream",
        lambda: "current-stream",
    )
    monkeypatch.setattr(rowwise_tbo.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        rowwise_tbo,
        "record_stream_event_no_compile",
        lambda stream: calls.append(("record_event", stream)) or event,  # type: ignore[func-returns-value]
    )

    class FakeDispatch:
        @staticmethod
        def apply(*args):
            calls.append(("dispatch", args))
            return torch.full((2, 3), 5.0)

    monkeypatch.setattr(rowwise_tbo, "_DispatchRowwiseAutograd", FakeDispatch)

    buffers = SimpleNamespace(dispatch_out=torch.zeros(2, 3))
    a_state = SimpleNamespace(
        lane_id=1,
        moe_inp=torch.ones(2, 3),
        dst_ranks=torch.zeros(2, 1, dtype=torch.long),
        dst_rows=torch.arange(2).view(2, 1),
        buffers=buffers,
        group_name="ep_group",
        rowwise_get_nblocks=61,
        rowwise_put_nblocks=62,
        rowwise_weighted_put_nblocks=63,
    )
    block = SimpleNamespace(block_idx=4, ep_pg="pg")

    d_state = rowwise_tbo.ep_no_sync_rowwise_tbo_stage_d_launch(block, a_state)  # type: ignore[arg-type]

    assert d_state.lane_id == 1
    assert d_state.a_state is a_state
    assert torch.equal(d_state.dispatch_out, torch.full((2, 3), 5.0))
    assert d_state.dispatch_done_event is event
    assert calls[0] == ("wait_stream", comm_stream, "current-stream")
    assert calls[1][0] == "dispatch"
    dispatch_args = calls[1][1]
    assert torch.equal(dispatch_args[0], a_state.moe_inp)
    assert dispatch_args[1] is None
    assert torch.equal(dispatch_args[2], a_state.dst_ranks)
    assert torch.equal(dispatch_args[3], a_state.dst_rows)
    assert dispatch_args[4] is buffers.dispatch_out
    assert dispatch_args[5:] == (None, "ep_group", "pg", 61, 62, False, True, True, True)
    assert calls[2] == ("record_event", comm_stream)
