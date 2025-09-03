import numpy as np
import pytest
import torch

from olmo_core.data.utils import (
    InstancePacker,
    SegmentTree,
    bucket_documents,
    get_cumulative_document_lengths,
    get_document_lengths,
    iter_batched,
    iter_document_indices,
    melt_batch,
    pack_documents_into_instances,
    segment_documents_into_instances,
    write_document_indices,
)


@pytest.mark.limit_memory("265 KB")
def test_segment_documents_into_instances(tmp_path):
    data = [1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 0] * 1000
    data_path = tmp_path / "data.npy"
    max_sequence_length = 4
    mmap = np.memmap(data_path, mode="w+", dtype=np.uint16, shape=(len(data),))
    indices_path = tmp_path / "indices.npy"
    mmap[:] = data
    mmap.flush()

    eos = 0
    dtype = np.uint16
    sample = (2, 42)

    results = []
    for _ in range(10):
        results.append(
            segment_documents_into_instances(
                path=data_path,
                target=indices_path,
                max_sequence_length=max_sequence_length,
                eos_token_id=eos,
                dtype=dtype,
                sample=sample,
            )
        )

    assert all([r[1] == 2 for r in results])


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

    # Should work when the instance BOS token is provided.
    bos_token_id = 50278
    assert get_document_lengths(
        torch.tensor(
            [
                bos_token_id,
                3,
                4,
                5,
                5,
                eos_token_id,
                bos_token_id,
                6,
                5,
                eos_token_id,
                bos_token_id,
                3,
                5,
                eos_token_id,
            ]
        ),
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
    ).tolist() == [6, 4, 4]

    # Should work when the instance ends with BOS token.
    bos_token_id = 50278
    assert get_document_lengths(
        torch.tensor(
            [bos_token_id, 3, 4, 5, 5, eos_token_id, bos_token_id, 6, 5, eos_token_id, bos_token_id]
        ),
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
    ).tolist() == [6, 4, 1]

    # Should work when the instance BOS token is provided and is same as EOS token.
    bos_token_id = eos_token_id
    assert get_document_lengths(
        torch.tensor(
            [
                bos_token_id,
                3,
                4,
                5,
                5,
                eos_token_id,
                bos_token_id,
                6,
                5,
                eos_token_id,
                bos_token_id,
                3,
                5,
                eos_token_id,
            ]
        ),
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
    ).tolist() == [6, 4, 4]


def test_document_lengths_with_min_document_length():
    eos_token_id = 50279
    chunk_size = 8
    tokens = [
        0,
        1,
        2,
        eos_token_id,  # should ignore this doc boundary
        4,
        5,
        eos_token_id,  # should ignore this doc boundary
        7,
        0,
        eos_token_id,  # should keep this! first one above chunk size
        2,
        3,
        4,
        5,
        6,
        eos_token_id,  # split here! this document is not above chunk size, but the next doc by itself.
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        eos_token_id,  # this is a long doc, gets combined with previous one for a super long doc!
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        eos_token_id,  # a document that is exactly at chunk size! keep
        3,  # this is the last boundary, so should keep it
    ]

    doc_lengths = get_document_lengths(
        torch.tensor(tokens, dtype=torch.int32),
        eos_token_id=eos_token_id,
        min_doc_length=chunk_size,
    )
    assert doc_lengths.tolist() == [10, 6, 11, 8, 1]


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


def test_segment_tree():
    seg_tree = SegmentTree(8)
    assert seg_tree.root_node.weight == 8

    leaf = seg_tree.query(8)  # leaf_id=7, weight=8
    assert leaf.leaf_id == 7
    assert leaf.weight == 8

    seg_tree.leaf_nodes[1].update(2)
    seg_tree.leaf_nodes[3].update(4)

    leaf = seg_tree.query(3)  # leaf_id=3, weight=4
    assert leaf.leaf_id == 3
    assert leaf.weight == 4


def test_instance_packer():
    # Follows the example from appendix (B) in https://arxiv.org/pdf/2404.10830
    packer = InstancePacker(8)
    assert packer._pack_document(0, 8) == 0
    assert packer._pack_document(1, 6) == 1
    assert packer._pack_document(2, 6) == 2
    assert packer._pack_document(3, 4) == 3
    assert packer._pack_document(4, 3) == 3

    # And here we extend the example...
    assert packer._pack_document(5, 2) == 1
    assert packer._pack_document(6, 3) == 4


def test_pack_documents_into_instances(tmp_path):
    data = [1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 0]
    data_path = tmp_path / "data.npy"
    mmap = np.memmap(data_path, dtype=np.uint16, mode="w+", shape=(len(data),))
    mmap[:] = data
    mmap.flush()

    instances, document_indices, total_tokens = pack_documents_into_instances(
        data_path, max_sequence_length=8, eos_token_id=0, dtype=np.uint16
    )
    assert instances == [[0], [1], [2], [3, 4]]
    assert document_indices[0].tolist() == [0, 8]
    assert total_tokens == len(data)
