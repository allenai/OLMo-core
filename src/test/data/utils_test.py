import numpy as np
import torch

from olmo_core.data.utils import (
    bucket_documents,
    get_cumulative_document_lengths,
    get_document_lengths,
    iter_batched,
    iter_document_indices,
    melt_batch,
    write_document_indices,
)


def test_iter_document_indices(tmp_path):
    data = [1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 0]
    data_path = tmp_path / "data.npy"
    mmap = np.memmap(data_path, mode="w+", dtype=np.uint16, shape=(len(data),))
    mmap[:] = data
    mmap.flush()
    write_document_indices(data_path, dtype=np.uint16, eos_token_id=0)
    assert list(
        iter_document_indices(data_path, eos_token_id=0, dtype=np.uint16, use_array_if_local=False)
    ) == [(0, 9), (9, len(data))]
    assert list(
        iter_document_indices(data_path, eos_token_id=0, dtype=np.uint16, use_array_if_local=True)
    ) == [(0, 9), (9, len(data))]


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


def test_iter_batched():
    instances = [
        {"input_ids": torch.tensor([1, 1, 1, 1])},
        {"input_ids": torch.tensor([2, 2, 2, 2])},
        {"input_ids": torch.tensor([3, 3])},
        {"input_ids": torch.tensor([4, 4])},
        {"input_ids": torch.tensor([5, 5])},
        {"input_ids": torch.tensor([6, 6])},
        {"input_ids": torch.tensor([7, 7])},
    ]
    batches = list(iter_batched(instances, 8))
    assert len(batches[0]) == 2
    assert len(batches[1]) == 4
    assert len(batches[2]) == 1


def test_bucket_documents(tmp_path):
    buckets = [2, 4]
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 0, 1, 2, 0])

    data_path = tmp_path / "data.npy"
    mmap = np.memmap(data_path, dtype=np.uint16, mode="w+", shape=data.shape)
    mmap[:] = data
    mmap.flush()

    write_document_indices(data_path, dtype=np.uint16, eos_token_id=0)

    n_og_docs, n_new_docs = bucket_documents(
        data_path, tmp_path / "buckets.npy", buckets=buckets, eos_token_id=0, dtype=np.uint16
    )
    assert n_og_docs == 3
    assert n_new_docs == 6

    buckets = (
        np.memmap(tmp_path / "buckets.npy", mode="r", dtype=np.uint32).reshape((-1, 2)).tolist()
    )
    assert buckets == [[0, 4], [4, 8], [8, 12], [13, 17], [17, 19], [19, 21]]
