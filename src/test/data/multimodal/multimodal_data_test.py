"""CPU tests for the Molmo2 stage-1 data pipeline: grounding format, branched-sequence
assembly, text-only handling, and the weighted mixture loader."""

import numpy as np

from olmo_core.data.multimodal import (
    MixtureDataLoader,
    MultimodalCollatorConfig,
    build_branched_sequence,
)
from olmo_core.data.multimodal.grounding import (
    format_points_tag,
    normalize_points,
    pointing_answer,
)

_SEQ = 8
_PATCH_DIM = 14 * 14 * 3


# ---------------------------------------------------------------------------
# Grounding / point format
# ---------------------------------------------------------------------------


def test_format_points_tag_html_v2():
    # Two points, normalized; expect 0-1000 3-digit coords, sorted by (x, y), image idx 1.
    pts = [[0.075, 0.812], [0.0, 0.5]]
    tag = format_points_tag(pts, "cat")
    assert tag == '<points coords="1 1 000 500 2 075 812">cat</points>'


def test_format_points_tag_sorted_and_clamped():
    pts = [[0.9, 0.1], [0.1, 0.9], [1.5, -0.2]]  # last is out of range -> clamped
    tag = format_points_tag(pts, "x")
    # sorted by (x, y): (0.1,0.9)->100 900, (0.9,0.1)->900 100, (1.0,0.0)->1000 000
    assert tag == '<points coords="1 1 100 900 2 900 100 3 1000 000">x</points>'


def test_pointing_answer_styles():
    pts = [[0.1, 0.2], [0.3, 0.4]]
    assert pointing_answer(pts, "cats", "pointing") == format_points_tag(pts, "cats")
    assert pointing_answer(pts, "cats", "point_count") == (
        f"Counting the {format_points_tag(pts, 'cats')} shows a total of 2."
    )
    assert pointing_answer([], "dogs", "pointing") == "There are none."
    assert pointing_answer([], "dogs", "point_count") == "There are none."


def test_normalize_points():
    # percent (point_scale=100) -> /100
    np.testing.assert_allclose(
        normalize_points(np.array([[50.0, 10.0]]), point_scale=100, image_size=None),
        [[0.5, 0.1]],
    )
    # pixel (point_scale=None) -> /image_size (w, h)
    np.testing.assert_allclose(
        normalize_points(np.array([[100.0, 50.0]]), point_scale=None, image_size=(200, 100)),
        [[0.5, 0.5]],
    )


# ---------------------------------------------------------------------------
# Branched (per-branch user turn) sequence assembly
# ---------------------------------------------------------------------------


def test_build_branched_sequence_two_branches():
    # prefix = BOS + 2 image tokens; 2 branches, each (user ctx 2 toks, answer 2 toks).
    out = build_branched_sequence(
        [100, 151938, 151937],
        [([10, 11], [20, 21]), ([12, 13], [30, 31])],
        eos_id=1,
    )
    assert out["input_ids"].tolist() == [100, 151938, 151937, 10, 11, 20, 21, 12, 13, 30, 31]
    # prefix 0,1,2 ; both branches start at position 3 (overlap, no carry-over)
    assert out["position_ids"].tolist() == [0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6]
    assert out["subsegment_ids"].tolist() == [10000, 10000, 10000, 0, 0, 0, 0, 1, 1, 1, 1]
    # loss only where a response (or its EOS) is predicted, scaled by 1/sqrt(2)
    nz = out["loss_masks"] > 0
    assert nz.tolist() == [False, False, False, False, True, True, True, False, True, True, True]
    np.testing.assert_allclose(out["loss_masks"][nz], 1.0 / np.sqrt(2), rtol=1e-3)
    # segment ends predict EOS
    assert out["labels"][6] == 1 and out["labels"][10] == 1


def test_build_branched_sequence_single():
    out = build_branched_sequence([100, 151938], [([10, 11], [20, 21])], eos_id=1)
    assert "subsegment_ids" not in out  # single branch -> no subsegments
    assert out["position_ids"].tolist() == [0, 1, 2, 3, 4, 5]  # sequential
    assert out["loss_masks"].tolist() == [0, 0, 0, 1, 1, 1]  # loss on response + its EOS target


# ---------------------------------------------------------------------------
# Mixture loader
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Tiny in-memory text-only dataset emitting the collator example dict."""

    def __init__(self, n: int, tag: int):
        self.n, self.tag = n, tag

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        L = 6
        return dict(
            input_ids=np.full(L, self.tag, dtype=np.int64),
            labels=np.full(L, -100, dtype=np.int64),
            loss_masks=np.ones(L, dtype=np.float32),
            position_ids=np.arange(L, dtype=np.int64),
            token_type_ids=np.zeros(L, dtype=np.int64),
            images=np.zeros((0, 729, _PATCH_DIM), dtype=np.float32),
            pooled_patches_idx=np.full((0, 4), -1, dtype=np.int64),
        )


def test_mixture_data_loader_weighted_sampling(tmp_path):
    ds = [_FakeDataset(1000, 10), _FakeDataset(500, 20), _FakeDataset(200, 30)]
    weights = [0.6, 0.3, 0.1]
    coll = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    dl = MixtureDataLoader(
        ds, weights, coll, work_dir=str(tmp_path), global_batch_size=4 * _SEQ, seed=0
    )
    dl.reshuffle(epoch=1)
    order = dl._order
    counts = np.bincount([s for s, _ in order], minlength=3) / len(order)
    np.testing.assert_allclose(counts, weights, atol=0.05)

    batch = next(iter(dl))
    assert tuple(batch["input_ids"].shape) == (4, _SEQ)
    # all sources here are text-only -> the batch still emits a single dummy zero crop
    # (with all-(-1) pooled indices) so the vision/connector path runs on every rank,
    # keeping FSDP collectives in lockstep. Nothing is spliced (no <im_patch> tokens).
    assert tuple(batch["images"].shape) == (4, 1, 729, _PATCH_DIM)
    assert (batch["pooled_patches_idx"] == -1).all()


def test_mixture_data_loader_normalizes_weights(tmp_path):
    ds = [_FakeDataset(10, 1), _FakeDataset(10, 2)]
    coll = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    dl = MixtureDataLoader(
        ds, [3.0, 1.0], coll, work_dir=str(tmp_path), global_batch_size=2 * _SEQ, seed=0
    )
    np.testing.assert_allclose(dl.weights, [0.75, 0.25])
