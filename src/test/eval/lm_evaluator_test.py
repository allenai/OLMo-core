import torch

from olmo_core.eval.lm_evaluator import LMEvaluator
from olmo_core.nn.attention.ring import (
    RingAttentionZigZagLoadBalancer,
    UlyssesLoadBalancer,
)


def _make_evaluator(labels=("c4", "wiki")) -> LMEvaluator:
    return LMEvaluator(
        name="test", batches=iter([]), labels=list(labels), device=torch.device("cpu")
    )


def test_update_metrics_with_logits_none():
    """update_metrics should work when logits is None (as with CP)."""
    evaluator = _make_evaluator(labels=("c4",))
    ce_loss = torch.tensor([[1.0, 2.0, 3.0]])
    batch = {"metadata": [{"label": "c4"}]}

    evaluator.update_metrics(batch, ce_loss=ce_loss, logits=None)

    metrics = evaluator.compute_metrics()
    assert "c4/CE loss" in metrics
    assert metrics["c4/CE loss"].item() == torch.tensor([1.0, 2.0, 3.0]).mean().item()


def test_update_metrics_skips_when_ce_loss_none():
    """update_metrics should be a no-op when ce_loss is None."""
    evaluator = _make_evaluator(labels=("c4",))
    batch = {"metadata": [{"label": "c4"}]}

    evaluator.update_metrics(batch, ce_loss=None, logits=None)

    # Metric was never updated with real data, so compute returns nan.
    metrics = evaluator.compute_metrics()
    assert torch.isnan(metrics["c4/CE loss"])


def test_update_metrics_with_label_mask():
    """update_metrics should apply label_mask to filter loss values."""
    evaluator = _make_evaluator(labels=("c4",))
    ce_loss = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    label_mask = torch.tensor([[True, False, True, False]])
    batch = {"metadata": [{"label": "c4"}], "label_mask": label_mask}

    evaluator.update_metrics(batch, ce_loss=ce_loss, logits=None)

    metrics = evaluator.compute_metrics()
    # Only positions 0 and 2 are selected: mean(1.0, 3.0) = 2.0
    assert metrics["c4/CE loss"].item() == 2.0


def test_update_metrics_with_sharded_ce_loss_and_label_mask():
    """Simulate CP: sharded ce_loss (B, T/CP) with matching sharded label_mask."""
    evaluator = _make_evaluator(labels=("c4",))
    # Simulate CP=2, rank 0: only sees first half of the sequence.
    sharded_ce_loss = torch.tensor([[1.0, 2.0]])
    sharded_label_mask = torch.tensor([[True, True]])
    batch = {"metadata": [{"label": "c4"}], "label_mask": sharded_label_mask}

    evaluator.update_metrics(batch, ce_loss=sharded_ce_loss, logits=None)

    metrics = evaluator.compute_metrics()
    assert metrics["c4/CE loss"].item() == 1.5


def test_update_metrics_multiple_labels():
    """update_metrics correctly routes losses to per-label metrics."""
    evaluator = _make_evaluator(labels=("c4", "wiki"))
    ce_loss = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
    batch = {"metadata": [{"label": "c4"}, {"label": "wiki"}]}

    evaluator.update_metrics(batch, ce_loss=ce_loss, logits=None)

    metrics = evaluator.compute_metrics()
    assert metrics["c4/CE loss"].item() == 1.5
    assert metrics["wiki/CE loss"].item() == 5.5


def test_label_mask_shard_zigzag():
    """batch_shard correctly shards a boolean label_mask (zig-zag strategy)."""
    lb = RingAttentionZigZagLoadBalancer(cp_rank=0, cp_world_size=2)
    # 8 tokens, CP=2 -> each rank gets 4 tokens (zig-zag: rank 0 gets chunks 0,3)
    label_mask = torch.tensor([[True, True, False, False, True, True, False, False]])
    (sharded,) = lb.batch_shard(inputs=[label_mask], seq_dims=[1], pad_values=[0])
    sharded = sharded.to(torch.bool)

    assert sharded.shape == (1, 4)
    # Zig-zag rank 0 gets tokens [0,1] (chunk 0) and [6,7] (chunk 3)
    assert sharded.tolist() == [[True, True, False, False]]


def test_label_mask_shard_ulysses():
    """batch_shard correctly shards a boolean label_mask (Ulysses strategy)."""
    lb = UlyssesLoadBalancer(cp_rank=0, cp_world_size=2)
    # 8 tokens, CP=2 -> rank 0 gets first 4 tokens (contiguous)
    label_mask = torch.tensor([[True, False, True, False, True, True, True, True]])
    (sharded,) = lb.batch_shard(inputs=[label_mask], seq_dims=[1], pad_values=[0])
    sharded = sharded.to(torch.bool)

    assert sharded.shape == (1, 4)
    assert sharded.tolist() == [[True, False, True, False]]


def test_label_mask_shard_with_padding():
    """batch_shard pads label_mask with False when sequence isn't evenly divisible."""
    lb = UlyssesLoadBalancer(cp_rank=1, cp_world_size=2)
    # 6 tokens, CP=2 -> pads to 6 (already divisible by 2), each rank gets 3
    label_mask = torch.tensor([[True, True, True, False, False, True]])
    (sharded,) = lb.batch_shard(inputs=[label_mask], seq_dims=[1], pad_values=[0])
    sharded = sharded.to(torch.bool)

    assert sharded.shape == (1, 3)
    # Rank 1 gets tokens [3,4,5]
    assert sharded.tolist() == [[False, False, True]]


def test_label_mask_shard_zigzag_with_padding():
    """batch_shard pads label_mask with False for zig-zag when padding is needed."""
    lb = RingAttentionZigZagLoadBalancer(cp_rank=0, cp_world_size=2)
    # 6 tokens with CP=2: zig-zag requires multiple of 4 (2*CP), so pads to 8
    label_mask = torch.tensor([[True, True, True, True, True, True]])
    (sharded,) = lb.batch_shard(inputs=[label_mask], seq_dims=[1], pad_values=[0])
    sharded = sharded.to(torch.bool)

    assert sharded.shape == (1, 4)
    # Zig-zag rank 0 gets chunks 0 and 3: tokens [0,1] and [6,7].
    # Tokens 6,7 are padded with 0 (False).
    assert sharded.tolist() == [[True, True, False, False]]
