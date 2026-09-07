"""The optional balanced splitter must preserve every tensor and metadata row."""

import pytest
import torch

from olmo_core.data.utils import split_batch, split_batch_balanced


@pytest.mark.parametrize(
    "count,maximum,expected",
    [
        (16, 3, [3, 3, 3, 3, 2, 2]),
        (32, 3, [3] * 10 + [2]),
        (16, 2, [2] * 8),
        (16, 4, [4] * 4),
        (8, 3, [3, 3, 2]),
        (1, 3, [1]),
    ],
)
def test_balanced_batch_preserves_data_and_gradients(count, maximum, expected):
    values = torch.arange(count * 7, dtype=torch.float64).reshape(count, 7).requires_grad_()
    batch = {"input_ids": values, "index": torch.arange(count), "metadata": list(range(count))}
    pieces = split_batch_balanced(batch, maximum)
    assert [part["input_ids"].shape[0] for part in pieces] == expected
    torch.testing.assert_close(torch.cat([p["input_ids"] for p in pieces]), values, rtol=0, atol=0)
    assert sum([p["metadata"] for p in pieces], []) == list(range(count))
    assert torch.cat([p["index"] for p in pieces]).tolist() == list(range(count))
    sum(p["input_ids"].square().sum() / values.numel() for p in pieces).backward()
    torch.testing.assert_close(values.grad, 2 * values / values.numel(), rtol=1e-14, atol=1e-14)
    if count % maximum == 0:
        original = split_batch(batch, maximum)
        for a, b in zip(pieces, original):
            torch.testing.assert_close(a["input_ids"], b["input_ids"], rtol=0, atol=0)


def test_balanced_batch_rejects_misaligned_metadata():
    batch = {"input_ids": torch.ones(4, 8), "metadata": ["missing rows"]}
    with pytest.raises(ValueError, match="instances"):
        split_batch_balanced(batch, 3)
    with pytest.raises(ValueError, match="positive"):
        split_batch_balanced(batch, 0)
