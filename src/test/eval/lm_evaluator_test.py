import torch

from olmo_core.eval.lm_evaluator import LMEvaluator


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


def test_compute_metrics_returns_ppl():
    """compute_metrics should return PPL = exp(CE loss) for each label."""
    evaluator = _make_evaluator(labels=("c4",))
    ce_loss = torch.tensor([[1.0, 2.0, 3.0]])
    batch = {"metadata": [{"label": "c4"}]}

    evaluator.update_metrics(batch, ce_loss=ce_loss, logits=None)

    metrics = evaluator.compute_metrics()
    expected_ce = torch.tensor([1.0, 2.0, 3.0]).mean()
    assert metrics["c4/PPL"].item() == torch.exp(expected_ce).item()


def test_reset_metrics():
    """reset_metrics should clear accumulated state so new updates start fresh."""
    evaluator = _make_evaluator(labels=("c4",))
    ce_loss = torch.tensor([[10.0, 20.0]])
    batch = {"metadata": [{"label": "c4"}]}

    evaluator.update_metrics(batch, ce_loss=ce_loss, logits=None)
    evaluator.reset_metrics()

    # After reset, no data has been seen, so compute returns nan.
    metrics = evaluator.compute_metrics()
    assert torch.isnan(metrics["c4/CE loss"])


def test_compute_metrics_unseen_label():
    """A label that receives no updates should produce NaN without affecting other labels."""
    evaluator = _make_evaluator(labels=("c4", "wiki"))
    ce_loss = torch.tensor([[1.0, 2.0]])
    batch = {"metadata": [{"label": "c4"}]}

    evaluator.update_metrics(batch, ce_loss=ce_loss, logits=None)

    metrics = evaluator.compute_metrics()
    assert metrics["c4/CE loss"].item() == 1.5
    assert torch.isnan(metrics["wiki/CE loss"])


def test_update_metrics_with_label_mask_batch_size_2():
    """label_mask is applied per-instance when batch size > 1."""
    evaluator = _make_evaluator(labels=("c4",))
    ce_loss = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    label_mask = torch.tensor([[True, False, False, True], [False, True, True, False]])
    batch = {"metadata": [{"label": "c4"}, {"label": "c4"}], "label_mask": label_mask}

    evaluator.update_metrics(batch, ce_loss=ce_loss, logits=None)

    metrics = evaluator.compute_metrics()
    # Instance 0: positions 0,3 selected -> [1.0, 4.0]
    # Instance 1: positions 1,2 selected -> [6.0, 7.0]
    # MeanMetric sees all 4 values: mean(1.0, 4.0, 6.0, 7.0) = 4.5
    assert metrics["c4/CE loss"].item() == 4.5
