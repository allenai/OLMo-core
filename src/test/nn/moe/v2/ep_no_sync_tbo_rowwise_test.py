from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import olmo_core.nn.ddp.block as block_mod
from olmo_core.nn.moe.v2 import ep_no_sync_tbo_rowwise as rowwise_tbo
from olmo_core.nn.moe.v2 import model as model_mod


def test_block_selects_rowwise_no_sync_tbo(monkeypatch):
    calls = []

    def fake_rowwise(block, x0, x1_ctx, x1_is_fresh, **kwargs):
        calls.append(("rowwise", block, x0, x1_ctx, x1_is_fresh, kwargs))
        return "rowwise-out", "rowwise-ctx"

    def fake_1d(*args, **kwargs):
        calls.append(("1d", args, kwargs))
        return "1d-out", "1d-ctx"

    monkeypatch.setattr(block_mod, "_combined_forward_ep_no_sync_tbo_rowwise", fake_rowwise)
    monkeypatch.setattr(block_mod, "_combined_forward_ep_no_sync_tbo", fake_1d)

    fake_block = SimpleNamespace(ep_no_sync_use_rowwise_all_to_all=True)
    out, ctx = block_mod.OLMoDDPTransformerBlock.combined_forward_ep_no_sync_tbo(
        fake_block,
        "x0",
        {"x1": "x1"},
        True,
        loss_div_factor=3.0,
    )

    assert (out, ctx) == ("rowwise-out", "rowwise-ctx")
    assert calls == [
        (
            "rowwise",
            fake_block,
            "x0",
            {"x1": "x1"},
            True,
            {"loss_div_factor": 3.0},
        )
    ]


def test_block_keeps_existing_1d_tbo_when_rowwise_disabled(monkeypatch):
    calls = []

    monkeypatch.setattr(
        block_mod,
        "_combined_forward_ep_no_sync_tbo_rowwise",
        lambda *args, **kwargs: calls.append(("rowwise", args, kwargs)),
    )

    def fake_1d(block, x0, x1_ctx, x1_is_fresh, **kwargs):
        calls.append(("1d", block, x0, x1_ctx, x1_is_fresh, kwargs))
        return "1d-out", "1d-ctx"

    monkeypatch.setattr(block_mod, "_combined_forward_ep_no_sync_tbo", fake_1d)

    fake_block = SimpleNamespace(ep_no_sync_use_rowwise_all_to_all=False)
    out, ctx = block_mod.OLMoDDPTransformerBlock.combined_forward_ep_no_sync_tbo(
        fake_block,
        "x0",
        {"x1": "x1"},
        True,
    )

    assert (out, ctx) == ("1d-out", "1d-ctx")
    assert calls[0][0] == "1d"


def test_rowwise_tbo_fails_closed_for_fp8():
    block = SimpleNamespace(
        ep_no_sync_use_rowwise_all_to_all=True,
        rowwise_fp8=SimpleNamespace(enabled=True),
    )

    with pytest.raises(NotImplementedError, match="Rowwise FP8"):
        rowwise_tbo._check_rowwise_tbo_supported(block)


def test_model_finalizes_rowwise_pending_context(monkeypatch):
    calls = []

    def fake_stage_c(block, ctx):
        calls.append(("stage_c", block, ctx))
        return "combined-pending"

    def fake_tail(block, ctx):
        calls.append(("tail", block, ctx))
        return "x1-final"

    monkeypatch.setattr(model_mod, "ep_no_sync_rowwise_tbo_stage_c_launch", fake_stage_c)
    monkeypatch.setattr(model_mod, "ep_no_sync_rowwise_tbo_stage_tail", fake_tail)

    class FakeModel:
        def maybe_forward_lm_head(self, x, lm_head_kwargs, labels=None):
            calls.append(("lm_head", x, lm_head_kwargs, labels))
            return f"head:{x}:{labels}"

    fake_block = SimpleNamespace()
    pending = rowwise_tbo._NoSyncRowwiseTboPendingContext(
        block=fake_block,
        lane_id=1,
        a_state=SimpleNamespace(),
        global_x_rank_major=torch.ones(1, 2),
    )

    h0, h1 = model_mod.MoEFusedV2Transformer._tbo_last_step(
        FakeModel(),
        "x0-final",
        pending,
        {"foo": "bar"},
        "labels0",
        "labels1",
    )

    assert h0 == "head:x0-final:labels0"
    assert h1 == "head:x1-final:labels1"
    assert calls == [
        ("stage_c", fake_block, pending),
        ("lm_head", "x0-final", {"foo": "bar"}, "labels0"),
        ("tail", fake_block, "combined-pending"),
        ("lm_head", "x1-final", {"foo": "bar"}, "labels1"),
    ]


def test_rowwise_tbo_combined_forward_fresh_schedule(monkeypatch):
    calls = []
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
            a_state=SimpleNamespace(lane_id=d_state.lane_id),
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
        fake_block,
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
    calls = []
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
        lambda stream: calls.append(("record_event", stream)) or event,
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
        rowwise_nblocks=64,
    )
    block = SimpleNamespace(block_idx=4, ep_pg="pg")

    d_state = rowwise_tbo.ep_no_sync_rowwise_tbo_stage_d_launch(block, a_state)

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
    assert dispatch_args[5:] == (None, "ep_group", "pg", 64, False, True, True, True)
    assert calls[2] == ("record_event", comm_stream)
