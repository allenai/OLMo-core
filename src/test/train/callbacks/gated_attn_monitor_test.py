"""Unit tests for gated attention monitor helpers."""

import math
from unittest.mock import Mock, PropertyMock

import pytest
import torch

from olmo_core.nn.attention import GateGranularity
from olmo_core.train.callbacks import GatedAttnMonitorCallback
from olmo_core.train.callbacks.gated_attn_monitor import (
    HeadGateStats,
    _parse_block_idx,
    per_head_gate_stats,
)
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module import TransformerTrainModule


def _logit(p: float) -> float:
    """Inverse sigmoid for constructing exact post-sigmoid gate values."""
    return math.log(p / (1.0 - p))


def test_parse_block_idx():
    assert _parse_block_idx("blocks.4.attention") == 4
    assert _parse_block_idx("blocks.19.attention.w_g") == 19
    assert _parse_block_idx("_fsdp_wrapped_module.blocks.9.attention") == 9
    assert _parse_block_idx("blocks.0.attention") == 0
    assert _parse_block_idx("model.blocks.14.attention") == 14
    assert _parse_block_idx("blocks.4.feed_forward") is None
    assert _parse_block_idx("blocks.4.attention_norm") is None
    assert _parse_block_idx("attention") is None


def test_per_head_gate_stats_headwise():
    n_heads = 4
    # Craft logits so post-sigmoid values are known exactly per head.
    # Head 0: all 0.5 -> mean 0.5, frac below thresholds = 0
    # Head 1: all 1e-4 -> mean 1e-4, both fracs = 1
    # Head 2: half 1e-4, half 0.5 -> mean ~(1e-4+0.5)/2, both fracs = 0.5
    # Head 3: all 5e-3 -> mean 5e-3, frac_1e2=1, frac_1e3=0
    B, T = 2, 3
    targets = torch.tensor([0.5, 1e-4, 0.5, 5e-3])  # default per head; head 2 overridden below
    logits = torch.empty(B, T, n_heads)
    for h, p in enumerate(targets.tolist()):
        logits[:, :, h] = _logit(p)
    # Head 2: alternate 1e-4 and 0.5 across the B*T=6 positions
    flat = logits[:, :, 2].reshape(-1)
    for i in range(flat.numel()):
        flat[i] = _logit(1e-4 if i % 2 == 0 else 0.5)

    stats = per_head_gate_stats(
        logits, n_heads=n_heads, head_dim=8, granularity=GateGranularity.headwise
    )
    assert len(stats) == n_heads
    assert all(s.count == B * T for s in stats)

    assert stats[0].mean == pytest.approx(0.5, abs=1e-5)
    assert stats[0].frac_lt_1e2 == pytest.approx(0.0)
    assert stats[0].frac_lt_1e3 == pytest.approx(0.0)

    assert stats[1].mean == pytest.approx(1e-4, abs=1e-6)
    assert stats[1].frac_lt_1e2 == pytest.approx(1.0)
    assert stats[1].frac_lt_1e3 == pytest.approx(1.0)

    assert stats[2].frac_lt_1e2 == pytest.approx(0.5)
    assert stats[2].frac_lt_1e3 == pytest.approx(0.5)
    assert stats[2].mean == pytest.approx((1e-4 + 0.5) / 2, abs=1e-5)

    assert stats[3].mean == pytest.approx(5e-3, abs=1e-6)
    assert stats[3].frac_lt_1e2 == pytest.approx(1.0)
    assert stats[3].frac_lt_1e3 == pytest.approx(0.0)


def test_per_head_gate_stats_elementwise():
    n_heads, head_dim = 2, 4
    B, T = 1, 2
    # Head 0: all elements 0.5; Head 1: all elements 1e-4
    probs = torch.empty(B, T, n_heads, head_dim)
    probs[:, :, 0, :] = 0.5
    probs[:, :, 1, :] = 1e-4
    logits = torch.logit(probs).reshape(B, T, n_heads * head_dim)

    stats = per_head_gate_stats(
        logits, n_heads=n_heads, head_dim=head_dim, granularity=GateGranularity.elementwise
    )
    assert len(stats) == n_heads
    assert all(s.count == B * T * head_dim for s in stats)

    assert stats[0].mean == pytest.approx(0.5, abs=1e-5)
    assert stats[0].frac_lt_1e2 == pytest.approx(0.0)
    assert stats[0].frac_lt_1e3 == pytest.approx(0.0)

    assert stats[1].mean == pytest.approx(1e-4, abs=1e-6)
    assert stats[1].frac_lt_1e2 == pytest.approx(1.0)
    assert stats[1].frac_lt_1e3 == pytest.approx(1.0)


def test_threshold_boundaries_are_strict():
    """Comparisons use strict ``<``; scores at/above the threshold do not count."""
    n_heads = 2
    logits = torch.empty(1, 4, n_heads)
    # Clear margin around the cut so logit/sigmoid round-trip cannot flip the side.
    logits[:, :, 0] = _logit(1.01e-2)
    logits[:, :, 1] = _logit(0.99e-2)

    stats = per_head_gate_stats(
        logits, n_heads=n_heads, head_dim=8, granularity=GateGranularity.headwise
    )
    assert stats[0].mean > 1e-2
    assert stats[0].frac_lt_1e2 == pytest.approx(0.0)
    assert stats[0].frac_lt_1e3 == pytest.approx(0.0)

    assert stats[1].mean < 1e-2
    assert stats[1].frac_lt_1e2 == pytest.approx(1.0)
    assert stats[1].frac_lt_1e3 == pytest.approx(0.0)

    logits_1e3 = torch.empty(1, 2, 1)
    logits_1e3[:, :, 0] = _logit(1.01e-3)
    above_stats = per_head_gate_stats(
        logits_1e3, n_heads=1, head_dim=8, granularity=GateGranularity.headwise
    )
    assert above_stats[0].mean > 1e-3
    assert above_stats[0].frac_lt_1e3 == pytest.approx(0.0)
    assert above_stats[0].frac_lt_1e2 == pytest.approx(1.0)

    logits_1e3[:, :, 0] = _logit(0.99e-3)
    below_stats = per_head_gate_stats(
        logits_1e3, n_heads=1, head_dim=8, granularity=GateGranularity.headwise
    )
    assert below_stats[0].mean < 1e-3
    assert below_stats[0].frac_lt_1e3 == pytest.approx(1.0)


def test_per_head_gate_stats_bfloat16_logits():
    """Hook path may see bf16 logits; stats are computed in fp32 after sigmoid."""
    n_heads = 2
    probs = torch.tensor([[[0.25, 0.75]]], dtype=torch.float32)  # (1, 1, 2)
    logits = torch.logit(probs).to(torch.bfloat16)

    stats = per_head_gate_stats(
        logits, n_heads=n_heads, head_dim=8, granularity=GateGranularity.headwise
    )
    assert stats[0].mean == pytest.approx(0.25, abs=1e-3)
    assert stats[1].mean == pytest.approx(0.75, abs=1e-3)
    assert stats[0].count == 1
    assert stats[1].count == 1


def test_microbatch_accumulation_matches_concat():
    """Updating across microbatches equals computing stats on the concatenated batch."""
    n_heads = 3
    mb0 = torch.empty(2, 4, n_heads)
    mb1 = torch.empty(1, 4, n_heads)
    for h, p in enumerate([0.2, 1e-4, 5e-3]):
        mb0[:, :, h] = _logit(p)
        mb1[:, :, h] = _logit(p)

    accum = [
        HeadGateStats(sum=0.0, count=0, count_lt_1e2=0, count_lt_1e3=0) for _ in range(n_heads)
    ]
    for mb in (mb0, mb1):
        for h, s in enumerate(
            per_head_gate_stats(
                mb, n_heads=n_heads, head_dim=8, granularity=GateGranularity.headwise
            )
        ):
            accum[h].update(s)

    cat_stats = per_head_gate_stats(
        torch.cat([mb0, mb1], dim=0),
        n_heads=n_heads,
        head_dim=8,
        granularity=GateGranularity.headwise,
    )
    for h in range(n_heads):
        assert accum[h].count == cat_stats[h].count
        assert accum[h].mean == pytest.approx(cat_stats[h].mean)
        assert accum[h].frac_lt_1e2 == pytest.approx(cat_stats[h].frac_lt_1e2)
        assert accum[h].frac_lt_1e3 == pytest.approx(cat_stats[h].frac_lt_1e3)


def test_head_gate_stats_update():
    a = HeadGateStats(sum=1.0, count=2, count_lt_1e2=1, count_lt_1e3=0)
    b = HeadGateStats(sum=3.0, count=2, count_lt_1e2=1, count_lt_1e3=1)
    a.update(b)
    assert a.sum == 4.0
    assert a.count == 4
    assert a.count_lt_1e2 == 2
    assert a.count_lt_1e3 == 1
    assert a.mean == pytest.approx(1.0)
    assert a.frac_lt_1e2 == pytest.approx(0.5)
    assert a.frac_lt_1e3 == pytest.approx(0.25)


def test_head_gate_stats_empty():
    empty = HeadGateStats(sum=0.0, count=0, count_lt_1e2=0, count_lt_1e3=0)
    assert empty.mean == 0.0
    assert empty.frac_lt_1e2 == 0.0
    assert empty.frac_lt_1e3 == 0.0


def test_layer_aggregate_from_per_head_stats():
    """Layer-level metrics are the element-weighted pool of per-head stats."""
    n_heads = 2
    B, T = 2, 2
    logits = torch.empty(B, T, n_heads)
    # Head 0: all 0.5 (frac=0); head 1: all 1e-4 (frac=1)
    logits[:, :, 0] = _logit(0.5)
    logits[:, :, 1] = _logit(1e-4)

    head_stats = per_head_gate_stats(
        logits, n_heads=n_heads, head_dim=8, granularity=GateGranularity.headwise
    )
    layer = HeadGateStats(sum=0.0, count=0, count_lt_1e2=0, count_lt_1e3=0)
    for s in head_stats:
        layer.update(s)

    assert layer.count == B * T * n_heads
    assert layer.mean == pytest.approx((0.5 + 1e-4) / 2, abs=1e-5)
    assert layer.frac_lt_1e2 == pytest.approx(0.5)
    assert layer.frac_lt_1e3 == pytest.approx(0.5)


def test_layer_aggregate_elementwise_uneven_head_activity():
    """Elementwise layer pool weights by head_dim elements, not by head count alone."""
    n_heads, head_dim = 2, 4
    B, T = 1, 1
    probs = torch.empty(B, T, n_heads, head_dim)
    # Head 0: one element below 1e-3, rest 0.5 -> 1/4 below both thresholds
    probs[:, :, 0, :] = 0.5
    probs[:, :, 0, 0] = 1e-4
    # Head 1: all 1e-4 -> all below
    probs[:, :, 1, :] = 1e-4
    logits = torch.logit(probs).reshape(B, T, n_heads * head_dim)

    head_stats = per_head_gate_stats(
        logits, n_heads=n_heads, head_dim=head_dim, granularity=GateGranularity.elementwise
    )
    layer = HeadGateStats(sum=0.0, count=0, count_lt_1e2=0, count_lt_1e3=0)
    for s in head_stats:
        layer.update(s)

    assert head_stats[0].frac_lt_1e3 == pytest.approx(0.25)
    assert head_stats[1].frac_lt_1e3 == pytest.approx(1.0)
    # 1 + 4 = 5 of 8 elements below 1e-3
    assert layer.count == 8
    assert layer.frac_lt_1e3 == pytest.approx(5 / 8)
    assert layer.frac_lt_1e2 == pytest.approx(5 / 8)


def test_per_head_gate_stats_shape_mismatch():
    logits = torch.zeros(2, 3, 8)
    with pytest.raises(ValueError, match="headwise"):
        per_head_gate_stats(
            logits, n_heads=4, head_dim=8, granularity=GateGranularity.headwise
        )
    with pytest.raises(ValueError, match="elementwise"):
        per_head_gate_stats(
            logits, n_heads=2, head_dim=8, granularity=GateGranularity.elementwise
        )


def test_callback_rejects_nonpositive_interval():
    callback = GatedAttnMonitorCallback(enabled=True, interval=0)
    callback._trainer = Mock()
    type(callback._trainer).train_module = PropertyMock(
        return_value=Mock(spec=TransformerTrainModule)
    )
    with pytest.raises(ValueError, match="interval"):
        callback.post_attach()


def test_callback_flush_records_head_and_layer_metrics():
    callback = GatedAttnMonitorCallback(enabled=True, interval=10)
    trainer = Mock()
    callback._trainer = trainer

    # Bypass the real Callback.step property (needs a live trainer.global_step).
    callback._dry_run_complete = True
    callback._layer_meta = {4: (2, 8, GateGranularity.headwise)}
    callback._accum = {
        4: [
            HeadGateStats(sum=1.0, count=2, count_lt_1e2=0, count_lt_1e3=0),  # mean 0.5
            HeadGateStats(sum=2e-4, count=2, count_lt_1e2=2, count_lt_1e3=2),  # mean 1e-4
        ]
    }

    callback._flush_metrics()

    recorded = {call.args[0]: call.args[1] for call in trainer.record_metric.call_args_list}
    assert set(recorded) == {
        "gated_attn/layer-4/head-0/1e-2",
        "gated_attn/layer-4/head-0/1e-3",
        "gated_attn/layer-4/head-0/mean",
        "gated_attn/layer-4/head-1/1e-2",
        "gated_attn/layer-4/head-1/1e-3",
        "gated_attn/layer-4/head-1/mean",
        "gated_attn/layer-4/1e-2",
        "gated_attn/layer-4/1e-3",
        "gated_attn/layer-4/mean",
    }
    assert recorded["gated_attn/layer-4/head-0/mean"] == pytest.approx(0.5)
    assert recorded["gated_attn/layer-4/head-1/mean"] == pytest.approx(1e-4)
    assert recorded["gated_attn/layer-4/1e-2"] == pytest.approx(0.5)
    assert recorded["gated_attn/layer-4/1e-3"] == pytest.approx(0.5)
    assert recorded["gated_attn/layer-4/mean"] == pytest.approx((0.5 + 1e-4) / 2)
    assert all(
        call.kwargs.get("reduce_type") == ReduceType.mean
        for call in trainer.record_metric.call_args_list
    )
    assert callback._accum is None


def test_callback_skips_flush_off_interval():
    callback = GatedAttnMonitorCallback(enabled=True, interval=10)
    trainer = Mock()
    trainer.global_step = 7
    callback._trainer = trainer
    callback._dry_run_complete = True
    callback._layer_meta = {4: (1, 8, GateGranularity.headwise)}
    callback._accum = {
        4: [HeadGateStats(sum=1.0, count=1, count_lt_1e2=0, count_lt_1e3=0)]
    }

    callback.pre_optim_step()
    trainer.record_metric.assert_not_called()
    assert callback._accum is not None
