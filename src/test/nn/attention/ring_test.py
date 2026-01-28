import torch

from olmo_core.nn.attention.ring import (
    RingAttentionZigZagLoadBalancer,
    UlyssesLoadBalancer,
)


def _get_zigzag_lb(rank: int, world_size: int) -> RingAttentionZigZagLoadBalancer:
    return RingAttentionZigZagLoadBalancer(cp_rank=rank, cp_world_size=world_size)


def test_zig_zag_load_balancer_padding():
    x, padding_added = _get_zigzag_lb(0, 4).pad(
        torch.tensor([0, 1, 2, 3, 4, 5]).unsqueeze(0), 1, -1
    )
    assert x.tolist() == [[0, 1, 2, 3, 4, 5, -1, -1]]
    assert padding_added == 2


def test_zig_zag_load_balancer_shard():
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).unsqueeze(0)
    assert _get_zigzag_lb(0, 4).batch_shard(inputs=[x], seq_dims=[1])[0].tolist() == [
        [
            0,
            7,
        ]
    ]
    assert _get_zigzag_lb(3, 4).batch_shard(inputs=[x], seq_dims=[1])[0].tolist() == [
        [
            3,
            4,
        ]
    ]


def test_zig_zag_load_balancer_shard_with_padding():
    x = torch.tensor([0, 1, 2, 3, 4, 5]).unsqueeze(0)
    assert _get_zigzag_lb(0, 4).batch_shard(inputs=[x], seq_dims=[1], pad_values=[-1])[
        0
    ].tolist() == [
        [
            0,
            -1,
        ]
    ]
    assert _get_zigzag_lb(3, 4).batch_shard(inputs=[x], seq_dims=[1], pad_values=[-1])[
        0
    ].tolist() == [
        [
            3,
            4,
        ]
    ]


def test_zig_zag_load_balancer_shard_by_document():
    x = torch.tensor(list(range(12))).unsqueeze(0)
    cu_doc_lens = torch.tensor([0, 8, 12])

    assert _get_zigzag_lb(0, 2).batch_shard_by_document(
        inputs=[x], seq_dims=[1], cu_doc_lens=cu_doc_lens
    )[0][0].tolist() == [
        [
            0,
            1,
            6,
            7,
            8,
            11,
        ]
    ]

    assert _get_zigzag_lb(1, 2).batch_shard_by_document(
        inputs=[x], seq_dims=[1], cu_doc_lens=cu_doc_lens
    )[0][0].tolist() == [
        [
            2,
            3,
            4,
            5,
            9,
            10,
        ]
    ]


def test_zig_zag_load_balancer_shard_by_document_with_padding():
    x = torch.tensor(list(range(12))).unsqueeze(0)
    cu_doc_lens = torch.tensor([0, 7, 10])

    res, opts = _get_zigzag_lb(0, 2).batch_shard_by_document(
        inputs=[x],
        seq_dims=[1],
        cu_doc_lens=cu_doc_lens,
        pad_values=[-1],
    )
    new_doc_lens = opts["cu_doc_lens"]
    assert new_doc_lens.tolist() == [0, 4, 6]
    assert res[0].tolist() == [
        [
            0,
            1,
            6,
            -1,
            7,
            -1,
        ]
    ]


def _get_ulysses_lb(rank: int, world_size: int) -> UlyssesLoadBalancer:
    return UlyssesLoadBalancer(cp_rank=rank, cp_world_size=world_size)


def test_ulysses_load_balancer_padding():
    x, padding_added = _get_ulysses_lb(0, 4).pad(
        torch.tensor([0, 1, 2, 3, 4, 5]).unsqueeze(0), 1, -1
    )
    assert x.tolist() == [[0, 1, 2, 3, 4, 5, -1, -1]]
    assert padding_added == 2


def test_ulysses_load_balancer_shard():
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).unsqueeze(0)
    # Ulysses uses contiguous sharding, so each rank gets a contiguous chunk
    assert _get_ulysses_lb(0, 4).batch_shard(inputs=[x], seq_dims=[1])[0].tolist() == [[0, 1]]
    assert _get_ulysses_lb(1, 4).batch_shard(inputs=[x], seq_dims=[1])[0].tolist() == [[2, 3]]
    assert _get_ulysses_lb(2, 4).batch_shard(inputs=[x], seq_dims=[1])[0].tolist() == [[4, 5]]
    assert _get_ulysses_lb(3, 4).batch_shard(inputs=[x], seq_dims=[1])[0].tolist() == [[6, 7]]


def test_ulysses_load_balancer_shard_with_padding():
    x = torch.tensor([0, 1, 2, 3, 4, 5]).unsqueeze(0)
    # 6 tokens with CP=4 -> pads to 8 tokens, then each rank gets 2
    assert _get_ulysses_lb(0, 4).batch_shard(inputs=[x], seq_dims=[1], pad_values=[-1])[
        0
    ].tolist() == [[0, 1]]
    assert _get_ulysses_lb(1, 4).batch_shard(inputs=[x], seq_dims=[1], pad_values=[-1])[
        0
    ].tolist() == [[2, 3]]
    assert _get_ulysses_lb(2, 4).batch_shard(inputs=[x], seq_dims=[1], pad_values=[-1])[
        0
    ].tolist() == [[4, 5]]
    assert _get_ulysses_lb(3, 4).batch_shard(inputs=[x], seq_dims=[1], pad_values=[-1])[
        0
    ].tolist() == [[-1, -1]]


def test_ulysses_load_balancer_shard_by_document():
    x = torch.tensor(list(range(12))).unsqueeze(0)
    cu_doc_lens = torch.tensor([0, 8, 12])

    # Ulysses with CP=2: rank 0 gets tokens 0-5, rank 1 gets tokens 6-11
    res0, opts0 = _get_ulysses_lb(0, 2).batch_shard_by_document(
        inputs=[x], seq_dims=[1], cu_doc_lens=cu_doc_lens
    )
    assert res0[0].tolist() == [[0, 1, 2, 3, 4, 5]]
    # Full sequences are reconstructed via all-to-all, so we pass through the original document lengths
    assert opts0["cu_doc_lens"].tolist() == cu_doc_lens.tolist()

    res1, opts1 = _get_ulysses_lb(1, 2).batch_shard_by_document(
        inputs=[x], seq_dims=[1], cu_doc_lens=cu_doc_lens
    )
    assert res1[0].tolist() == [[6, 7, 8, 9, 10, 11]]
    # Full sequences are reconstructed via all-to-all, so we pass through the original document lengths
    assert opts1["cu_doc_lens"].tolist() == cu_doc_lens.tolist()


def test_ulysses_load_balancer_shard_by_document_with_padding():
    # 10 tokens with CP=4 -> pads to 12 tokens (next multiple of 4), then each rank gets 3
    x = torch.tensor(list(range(10))).unsqueeze(0)
    cu_doc_lens = torch.tensor([0, 6, 10])
    # Padding adds 2 tokens, creating a synthetic document: [0, 6, 10] -> [0, 6, 10, 12]
    expected_cu_doc_lens = [0, 6, 10, 12]

    res0, opts0 = _get_ulysses_lb(0, 4).batch_shard_by_document(
        inputs=[x], seq_dims=[1], cu_doc_lens=cu_doc_lens, pad_values=[-1]
    )
    assert res0[0].tolist() == [[0, 1, 2]]
    # cu_doc_lens should include the padding as a synthetic document
    assert opts0["cu_doc_lens"].tolist() == expected_cu_doc_lens
    assert opts0["max_doc_len"] == 6  # max of doc lengths: 6, 4, 2

    res1, opts1 = _get_ulysses_lb(1, 4).batch_shard_by_document(
        inputs=[x], seq_dims=[1], cu_doc_lens=cu_doc_lens, pad_values=[-1]
    )
    assert res1[0].tolist() == [[3, 4, 5]]
    assert opts1["cu_doc_lens"].tolist() == expected_cu_doc_lens

    res2, opts2 = _get_ulysses_lb(2, 4).batch_shard_by_document(
        inputs=[x], seq_dims=[1], cu_doc_lens=cu_doc_lens, pad_values=[-1]
    )
    assert res2[0].tolist() == [[6, 7, 8]]
    assert opts2["cu_doc_lens"].tolist() == expected_cu_doc_lens

    res3, opts3 = _get_ulysses_lb(3, 4).batch_shard_by_document(
        inputs=[x], seq_dims=[1], cu_doc_lens=cu_doc_lens, pad_values=[-1]
    )
    # Last rank gets token 9 plus 2 padding tokens
    assert res3[0].tolist() == [[9, -1, -1]]
    assert opts3["cu_doc_lens"].tolist() == expected_cu_doc_lens
