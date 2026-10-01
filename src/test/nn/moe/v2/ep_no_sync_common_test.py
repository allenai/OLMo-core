import torch

from olmo_core.nn.moe.v2.ep_no_sync_common import build_keep_reorder


def _ref_keep_reorder(requested: list[int], keep: list[int]):
    """Independent reference: stably pack kept-then-dropped rows for per-expert capacity drop."""
    kept: list[int] = []
    dropped: list[int] = []
    tok = 0
    for r, k in zip(requested, keep):
        for pos in range(r):
            (kept if pos < k else dropped).append(tok)
            tok += 1
    packed = kept + dropped
    inverse = [0] * tok
    for slot, original in enumerate(packed):
        inverse[original] = slot
    keep_mask = [slot < len(kept) for slot in range(tok)]
    return packed, inverse, keep_mask


def _check(requested: list[int], keep: list[int]):
    num_out = sum(requested)
    reorder, inverse, packed_keep_mask = build_keep_reorder(
        torch.tensor(requested, dtype=torch.long),
        torch.tensor(keep, dtype=torch.long),
        num_out,
    )
    ref_reorder, ref_inverse, ref_keep_mask = _ref_keep_reorder(requested, keep)
    assert reorder.tolist() == ref_reorder
    assert inverse.tolist() == ref_inverse
    assert packed_keep_mask.tolist() == ref_keep_mask
    # reorder and inverse must be inverse permutations of each other.
    assert reorder.index_select(0, inverse).tolist() == list(range(num_out))


def test_build_keep_reorder_partial_drop():
    _check(requested=[3, 2, 4], keep=[2, 2, 1])


def test_build_keep_reorder_no_drop():
    _check(requested=[3, 2, 4], keep=[3, 2, 4])


def test_build_keep_reorder_full_drop_some_experts():
    _check(requested=[2, 3, 1], keep=[0, 3, 0])
