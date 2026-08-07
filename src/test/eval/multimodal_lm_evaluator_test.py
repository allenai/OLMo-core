import torch

from olmo_core.eval import MultimodalLMEvaluator


def _make_evaluator() -> MultimodalLMEvaluator:
    return MultimodalLMEvaluator(
        name="pixmo-cap-validation",
        batches=iter([]),
        device=torch.device("cpu"),
    )


def test_multimodal_lm_evaluator_normalizes_summed_weighted_loss():
    evaluator = _make_evaluator()
    batch = {
        "labels": torch.tensor([[1, 2, 3, -100]]),
        "loss_masks": torch.tensor([[0.0, 1.0, 0.5, 1.0]]),
    }
    # Weighted token losses: 2 * 1.0 + 4 * 0.5 = 4.0. The ignored final label
    # contributes neither loss nor denominator weight.
    evaluator.update_metrics(batch, ce_loss=torch.tensor(4.0), logits=None)

    metrics = evaluator.compute_metrics()

    torch.testing.assert_close(metrics["CE loss"], torch.tensor(4.0 / 1.5))
    torch.testing.assert_close(metrics["PPL"], torch.exp(torch.tensor(4.0 / 1.5)))


def test_multimodal_lm_evaluator_accumulates_by_loss_weight():
    evaluator = _make_evaluator()
    evaluator.update_metrics(
        {"labels": torch.tensor([[1]]), "loss_masks": torch.tensor([[1.0]])},
        ce_loss=torch.tensor(2.0),
        logits=None,
    )
    evaluator.update_metrics(
        {"labels": torch.tensor([[1]]), "loss_masks": torch.tensor([[3.0]])},
        ce_loss=torch.tensor(12.0),
        logits=None,
    )

    metrics = evaluator.compute_metrics()

    torch.testing.assert_close(metrics["CE loss"], torch.tensor(3.5))
