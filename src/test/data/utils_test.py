import torch

from olmo_core.data.utils import (
    get_cumulative_document_lengths,
    get_document_lengths,
    melt_batch,
)


def test_melt_batch():
    batch = {
        "input_ids": torch.randint(0, 32, (2, 12)),
        "index": torch.tensor([0, 1]),
        "metadata": [{"path": "x"}, {"path": "y"}],
    }
    new_batch = melt_batch(batch, 4)
    assert new_batch.keys() == batch.keys()
    assert new_batch["input_ids"].shape == (6, 4)
    assert new_batch["index"].tolist() == [0, 0, 0, 1, 1, 1]
    assert new_batch["metadata"] == [{"path": "x"}] * 3 + [{"path": "y"}] * 3


def test_get_document_lengths():
    eos_token_id = 50279

    # Should work when the instance starts with EOS token.
    assert get_document_lengths(
        torch.tensor([eos_token_id, 3, 4, 5, 5, eos_token_id, 6, 5, eos_token_id, 3, 5]),
        eos_token_id=eos_token_id,
    ).tolist() == [1, 5, 3, 2]

    # Should work when the instance ends with EOS token.
    assert get_document_lengths(
        torch.tensor([3, 4, 5, 5, eos_token_id, 6, 5, eos_token_id, 3, 5, eos_token_id]),
        eos_token_id=eos_token_id,
    ).tolist() == [5, 3, 3]


def test_get_cumulative_document_lengths():
    assert get_cumulative_document_lengths(
        torch.tensor(
            [
                [1, 5, 3, 2, 0],
                [5, 3, 3, 0, 0],
            ],
            dtype=torch.int32,
        )
    ).tolist() == [0, 1, 6, 9, 11, 16, 19, 22]
