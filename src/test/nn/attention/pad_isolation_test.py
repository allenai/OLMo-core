"""Pad positions must not attend to each other.

The collator marks every pad position with the same ``example_ids`` sentinel (-1). Under
the packed-example rule ``example_id[q] == example_id[kv]`` that made the whole pad tail
one mutually-visible causal segment, so FlexAttention computed it in full -- at the
shipped single-image geometry (~4 examples in a 16,384 slot) that was ~92% of all
computed blocks, spent on tokens whose outputs are discarded.
"""

import torch

from olmo_core.nn.attention.backend import FlexAttentionBackend

_S = 2048
_DEV = torch.device("cpu")


def _example_ids(n_examples: int, tokens_each: int) -> torch.Tensor:
    eid = torch.full((1, _S), -1, dtype=torch.long)
    pos = 0
    for i in range(n_examples):
        eid[0, pos : pos + tokens_each] = i
        pos += tokens_each
    return eid


def _computed_blocks(eid: torch.Tensor) -> int:
    bm = FlexAttentionBackend.build_block_mask_from_vectors(1, _S, _DEV, example_id=eid)
    full = int(bm.kv_num_blocks.sum())
    partial = int(bm.full_kv_num_blocks.sum()) if bm.full_kv_num_blocks is not None else 0
    return full + partial


def _mask_mod(eid: torch.Tensor):
    backend = FlexAttentionBackend(head_dim=64, n_heads=4)
    return backend._build_mask_mod(None, None, None, eid)


def test_pad_does_not_attend_to_other_pad():
    eid = _example_ids(2, 256)  # 512 real tokens, 1536 pad
    mod = _mask_mod(eid)
    b, h = torch.tensor(0), torch.tensor(0)
    pad_q, other_pad = torch.tensor(1500), torch.tensor(1000)

    assert not bool(mod(b, h, pad_q, other_pad))
    # ...but the row is not empty: a pad position still sees itself, so softmax is defined.
    assert bool(mod(b, h, pad_q, pad_q))


def test_pad_and_real_stay_mutually_invisible():
    eid = _example_ids(2, 256)
    mod = _mask_mod(eid)
    b, h = torch.tensor(0), torch.tensor(0)
    real_q, pad_kv = torch.tensor(300), torch.tensor(1500)
    pad_q, real_kv = torch.tensor(1500), torch.tensor(300)

    assert not bool(mod(b, h, real_q, pad_kv))
    assert not bool(mod(b, h, pad_q, real_kv))


def test_real_example_attention_is_unchanged():
    """Within an example: causal. Across examples: blocked. Neither involves the pad rule."""
    eid = _example_ids(2, 256)
    mod = _mask_mod(eid)
    b, h = torch.tensor(0), torch.tensor(0)

    # causal inside example 0
    assert bool(mod(b, h, torch.tensor(200), torch.tensor(100)))
    assert not bool(mod(b, h, torch.tensor(100), torch.tensor(200)))
    # example 1 cannot see example 0
    assert not bool(mod(b, h, torch.tensor(300), torch.tensor(100)))
    # causal inside example 1
    assert bool(mod(b, h, torch.tensor(400), torch.tensor(300)))


def test_sparse_pack_costs_far_fewer_blocks_than_it_did():
    """A mostly-empty pack must not cost a full causal triangle over its pad tail."""
    sparse = _computed_blocks(_example_ids(2, 256))  # 512 of 2048 real
    dense = _computed_blocks(_example_ids(8, 256))  # 2048 of 2048 real

    # With the pad tail collapsed to a diagonal, a 25%-full pack is far cheaper than a
    # full one. Before the fix the sparse pack cost *more* blocks than the dense one,
    # because its pad tail was a single large causal segment.
    assert sparse < dense
