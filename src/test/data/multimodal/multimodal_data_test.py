"""CPU tests for the Molmo2 stage-1 data pipeline: grounding format, branched-sequence
assembly, text-only handling, and the weighted mixture loader."""

import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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
from olmo_core.data.multimodal.native_text_replay import NativeTextReplayDataset
from olmo_core.data.multimodal.packing import (
    _select_buffered_pack_indices,
    greedy_pack_indices,
    iter_packs,
    pack_examples,
)
from olmo_core.data.multimodal.prefetch import prefetch_map
from olmo_core.data.multimodal.rng import make_random_state
from olmo_core.exceptions import OLMoConfigurationError

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


class _CountingFakeDataset(_FakeDataset):
    """Fake dataset that makes each ref observable and counts preprocessing calls."""

    def __init__(self, n: int, tag: int):
        super().__init__(n, tag)
        self.loads = 0

    def __getitem__(self, i):
        self.loads += 1
        out = super().__getitem__(i)
        out["input_ids"] = np.full(len(out["input_ids"]), self.tag + i, dtype=np.int64)
        return out


class _FingerprintedFakeDataset(_FakeDataset):
    """Fake source implementing the preferred content-fingerprint protocol."""

    content_fingerprint_version = "fake-content-v1"

    def __init__(self, n: int, tag: int, content_fingerprint: str):
        super().__init__(n, tag)
        self.content_fingerprint = content_fingerprint
        # Ensure the explicit content protocol wins over the backwards-compatible fallback.
        self.fingerprint = "must-not-be-used"


class _FailingFakeDataset(_CountingFakeDataset):
    def __init__(self, n: int, tag: int):
        super().__init__(n, tag)
        self.fail_indices = set()

    def __getitem__(self, i):
        if i in self.fail_indices:
            self.loads += 1
            raise ValueError(f"synthetic failure for {i}")
        return super().__getitem__(i)


class _EpochAwareFakeDataset(_FakeDataset):
    def __init__(self, n: int, tag: int):
        super().__init__(n, tag)
        self.requests = []

    def get(self, i: int, epoch: int):
        self.requests.append((i, epoch))
        return super().__getitem__(i)


def test_mixture_data_loader_weighted_sampling(tmp_path):
    ds = [_FakeDataset(1000, 10), _FakeDataset(500, 20), _FakeDataset(200, 30)]
    weights = [0.6, 0.3, 0.1]
    coll = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    dl = MixtureDataLoader(
        ds, weights, coll, work_dir=str(tmp_path), global_batch_size=4 * _SEQ, seed=0
    )
    dl.reshuffle(epoch=1)
    order = dl._order
    counts = np.bincount([source for source, _, _ in order], minlength=3) / len(order)
    np.testing.assert_allclose(counts, weights, atol=0.05)

    batch = next(iter(dl))
    assert tuple(batch["input_ids"].shape) == (4, _SEQ)
    # all sources here are text-only -> the batch still emits a single dummy zero crop
    # (with all-(-1) pooled indices) so the vision/connector path runs on every rank,
    # keeping FSDP collectives in lockstep. Nothing is spliced (no <im_patch> tokens).
    assert tuple(batch["images"].shape) == (4, 1, 729, _PATCH_DIM)
    assert (batch["pooled_patches_idx"] == -1).all()


def test_mixture_data_loader_tracks_source_epoch_for_augmentation(tmp_path):
    dataset = _EpochAwareFakeDataset(2, 10)
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    loader = MixtureDataLoader(
        [dataset],
        [1.0],
        collator,
        work_dir=str(tmp_path),
        global_batch_size=2 * _SEQ,
        epoch_instances=6,
        seed=3,
    )
    loader.reshuffle(epoch=1)

    assert [source_epoch for _, _, source_epoch in loader._order] == [0, 0, 1, 1, 2, 2]
    for ref in loader._order:
        loader._try_load_example(ref)
    assert [epoch for _, epoch in dataset.requests] == [0, 0, 1, 1, 2, 2]


# ---------------------------------------------------------------------------
# Sequence packing
# ---------------------------------------------------------------------------


def _img_example(n_text: int, n_crops: int, tag: int):
    L = n_crops + n_text  # n_crops <im_patch> tokens (id 1) + text
    return dict(
        input_ids=np.array([1] * n_crops + [tag] * n_text, dtype=np.int64),
        labels=np.full(L, tag, dtype=np.int64),
        loss_masks=np.ones(L, dtype=np.float32),
        position_ids=np.arange(L, dtype=np.int64),
        token_type_ids=np.array([1] * n_crops + [0] * n_text, dtype=np.int64),
        images=np.full((n_crops, 729, _PATCH_DIM), float(tag), dtype=np.float32),
        pooled_patches_idx=np.arange(n_crops * 4).reshape(n_crops, 4).astype(np.int64),
    )


def _text_example(n_text: int = 4, tag: int = 3):
    return dict(
        input_ids=np.full(n_text, tag, dtype=np.int64),
        labels=np.full(n_text, tag, dtype=np.int64),
        loss_masks=np.ones(n_text, dtype=np.float32),
        position_ids=np.arange(n_text, dtype=np.int64),
        token_type_ids=np.zeros(n_text, dtype=np.int64),
        images=np.zeros((0, 729, _PATCH_DIM), dtype=np.float32),
        pooled_patches_idx=np.full((0, 4), -1, dtype=np.int64),
    )


def test_multimodal_collator_marks_only_retained_real_tokens_for_routing():
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=8).build()
    batch = collator([_text_example(n_text=3), _text_example(n_text=10)])

    assert batch["router_token_mask"].dtype == torch.bool
    assert batch["router_token_mask"].tolist() == [
        [True, True, True, False, False, False, False, False],
        [True, True, True, True, True, True, True, True],
    ]
    assert batch["image_crop_counts"].tolist() == [0, 0]
    assert batch["pooled_token_counts"].tolist() == [0, 0]


def test_greedy_pack_indices():
    # next-fit: [3,3]->ok(6), +5 overflows 8 -> new group, +2 fits(7)
    assert greedy_pack_indices([3, 3, 5, 2], seq_len=8) == [[0, 1], [2, 3]]
    assert greedy_pack_indices([10], seq_len=8) == [[0]]  # over-length example alone


def test_greedy_pack_indices_crop_budget():
    assert greedy_pack_indices(
        [4, 4, 4, 4],
        seq_len=20,
        crop_counts=[1, 1, 1, 1],
        max_crops_per_pack=2,
    ) == [[0, 1], [2, 3]]


def test_iter_packs_crop_budget():
    examples = [_img_example(n_text=2, n_crops=2, tag=i) for i in range(4)]
    packs = list(iter_packs(examples, seq_len=32, max_crops_per_pack=4))
    assert len(packs) == 2
    assert [pack["images"].shape[0] for pack in packs] == [4, 4]


def test_iter_packs_keeps_image_and_text_examples_separate():
    image = _img_example(n_text=2, n_crops=1, tag=5)
    text = _text_example(n_text=3, tag=7)
    packs = list(iter_packs([image, text, image], seq_len=32))
    assert len(packs) == 3
    assert packs[0]["images"].shape[0] == 1
    assert packs[1]["images"].shape[0] == 0
    assert packs[2]["images"].shape[0] == 1


def test_iter_packs_buffered_solver_improves_fill_and_flushes():
    examples = [_text_example(n_text=n, tag=i + 2) for i, n in enumerate([6, 6, 4, 4])]
    packs = list(
        iter_packs(
            examples,
            seq_len=10,
            max_crops_per_pack=1,
            buffer_size=4,
        )
    )

    assert [len(pack["input_ids"]) for pack in packs] == [10, 10]
    assert sum(len(np.unique(pack["example_ids"])) for pack in packs) == len(examples)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([pack["input_ids"] for pack in packs])),
        np.sort(np.concatenate([example["input_ids"] for example in examples])),
    )


def test_buffered_solver_uses_stage_specific_image_weight():
    lengths = [10, 1]
    crops = [0, 2]

    assert _select_buffered_pack_indices(lengths, crops, 10, 2, image_weight=1.0) == [0]
    assert _select_buffered_pack_indices(lengths, crops, 10, 2, image_weight=30.0) == [1]


def test_iter_packs_buffered_solver_never_exceeds_unaligned_sequence_length():
    packs = list(
        iter_packs(
            [_text_example(n_text=1024), _text_example(n_text=2)],
            seq_len=1025,
            max_crops_per_pack=1,
            buffer_size=2,
        )
    )
    assert [len(pack["input_ids"]) for pack in packs] == [1024, 2]


def test_iter_packs_buffered_solver_respects_crop_budget_and_mixes_modalities():
    image_a = _img_example(n_text=2, n_crops=2, tag=5)
    text = _text_example(n_text=6, tag=7)
    image_b = _img_example(n_text=2, n_crops=2, tag=9)
    image_a["_source_name"] = "image-a"
    text["_source_name"] = "text"
    image_b["_source_name"] = "image-b"

    packs = list(
        iter_packs(
            [image_a, text, image_b],
            seq_len=10,
            max_crops_per_pack=2,
            buffer_size=3,
        )
    )

    assert all(len(pack["input_ids"]) <= 10 for pack in packs)
    assert all(pack["images"].shape[0] <= 2 for pack in packs)
    assert any(set(pack["pack_source_names"]) == {"image-a", "text"} for pack in packs)


def test_tulu_conversation_is_bounded_by_sequence_length():
    from olmo_core.data.multimodal.tulu import Tulu4Dataset, Tulu4DatasetConfig

    dataset = object.__new__(Tulu4Dataset)
    dataset.config = Tulu4DatasetConfig(max_sequence_length=40, message_format="document")
    dataset.tokenizer = _FakeTok()
    dataset._data = [
        {
            "messages": [
                {"role": "user", "content": "a long user message"},
                {"role": "assistant", "content": "a long assistant response"},
            ]
        }
    ]
    example = dataset[0]
    assert len(example["input_ids"]) == 40
    assert np.count_nonzero(example["loss_masks"]) > 0
    assert all(
        value.ndim != 1 or len(value) <= 40
        for value in example.values()
        if isinstance(value, np.ndarray)
    )


def test_tulu_filter_uses_process_local_indices(tmp_path, monkeypatch):
    from datasets import Dataset

    import olmo_core.data.multimodal.tulu as tulu_module
    from olmo_core.data.multimodal.tulu import Tulu4Dataset, Tulu4DatasetConfig

    data_path = tmp_path / "tulu"
    Dataset.from_dict(
        {
            "row_id": [0, 1, 2, 3],
            "category": ["general", "code", "general", "general"],
            "source": ["keep", "keep", "keep", "allenai/hardcoded-olmo"],
            "first_message_qwen3_tokens": [100, 100, 2305, 100],
            "empty_messages": [False, False, False, False],
            "has_special_token": [False, False, False, False],
        }
    ).save_to_disk(str(data_path))
    monkeypatch.setattr(tulu_module, "TULU4_DATA", str(data_path))

    dataset = object.__new__(Tulu4Dataset)
    dataset.config = Tulu4DatasetConfig(max_first_msg_len=2304)

    first = dataset._load_filtered()
    second = dataset._load_filtered()

    assert first["row_id"] == second["row_id"] == [0]
    assert type(first._indices).__name__ == "InMemoryTable"
    assert type(second._indices).__name__ == "InMemoryTable"
    assert not list(data_path.glob("cache-*.arrow"))


def test_tulu_binary_response_loss_weighting():
    from olmo_core.data.multimodal.tulu import Tulu4Dataset, Tulu4DatasetConfig

    dataset = object.__new__(Tulu4Dataset)
    dataset.config = Tulu4DatasetConfig(
        max_sequence_length=64,
        loss_token_weighting="none",
        message_format="document",
    )
    dataset.tokenizer = _FakeTok()
    dataset._data = [
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    ]

    positive = dataset[0]["loss_masks"]
    positive = positive[positive > 0]
    np.testing.assert_array_equal(positive, np.ones_like(positive))


def test_tulu_uses_s002_document_boundaries_without_chat_headers():
    from olmo_core.data.multimodal.tulu import Tulu4Dataset, Tulu4DatasetConfig

    dataset = object.__new__(Tulu4Dataset)
    dataset.config = Tulu4DatasetConfig(
        max_sequence_length=128,
        loss_token_weighting="none",
        message_format="document",
    )
    dataset.tokenizer = _FakeTok()
    dataset._data = [
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    ]

    example = dataset.get(0, 0)
    assert example["input_ids"][0] == dataset.tokenizer.eos_token_id
    assert example["labels"][-1] == dataset.tokenizer.eos_token_id
    assert example["loss_masks"][-1] == 1
    assert dataset.tokenizer.prompts[0].startswith("text_sft")
    assert all("<|im_start|>" not in prompt for prompt in dataset.tokenizer.prompts)


def test_pack_examples_concat_and_offsets():
    a = _img_example(n_text=3, n_crops=1, tag=5)  # len 4, 1 crop
    b = _img_example(n_text=2, n_crops=2, tag=7)  # len 4, 2 crops
    a["_source_name"] = "pixmo_points_train"
    b["_source_name"] = "pixmo_count_train"
    packed = pack_examples([a, b])

    assert packed["pack_source_names"] == ["pixmo_points_train", "pixmo_count_train"]

    assert packed["input_ids"].tolist() == [1, 5, 5, 5, 1, 1, 7, 7]
    assert packed["position_ids"].tolist() == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
    ]  # positions reset per example
    assert packed["example_ids"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    # images concatenated along the crop axis (1 + 2 = 3 crops)
    assert packed["images"].shape == (3, 729, _PATCH_DIM)
    # b's pooled indices are offset by a's crop-patch count (1 crop * 729 patches)
    np.testing.assert_array_equal(packed["pooled_patches_idx"][0], [0, 1, 2, 3])  # a
    np.testing.assert_array_equal(packed["pooled_patches_idx"][1], np.arange(4) + 729)  # b crop 0
    np.testing.assert_array_equal(
        packed["pooled_patches_idx"][2], np.arange(4, 8) + 729
    )  # b crop 1


def test_prefetch_map_order_and_completeness():
    import time

    def slow(x):
        time.sleep(0.001 * ((x * 7) % 5))  # uneven work so threads finish out of order
        return x * x

    items = list(range(50))
    for workers in (0, 1, 4):
        out = list(prefetch_map(slow, iter(items), num_workers=workers, max_in_flight=8))
        assert out == [x * x for x in items]  # order preserved, nothing dropped


def test_prefetch_does_not_change_packing():
    # Packing output must be identical whether examples are loaded sync or prefetched.
    exs = [_img_example(n_text=3 + (i % 4), n_crops=1, tag=i + 2) for i in range(12)]
    get = lambda i: exs[i]  # noqa: E731
    sync = list(iter_packs((get(i) for i in range(12)), seq_len=12))
    pref = list(iter_packs(prefetch_map(get, iter(range(12)), num_workers=4), seq_len=12))
    assert len(sync) == len(pref)
    for a, b in zip(sync, pref):
        np.testing.assert_array_equal(a["input_ids"], b["input_ids"])
        np.testing.assert_array_equal(a["example_ids"], b["example_ids"])


def test_mixture_data_loader_packs(tmp_path):
    ds = [_FakeDataset(200, 10), _FakeDataset(100, 20)]
    coll = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    dl = MixtureDataLoader(
        ds, [0.5, 0.5], coll, work_dir=str(tmp_path), global_batch_size=2 * _SEQ, seed=0, pack=True
    )
    dl.reshuffle(epoch=1)
    batch = next(iter(dl))
    # _FakeDataset emits length-6 text-only examples; with _SEQ=8 only one fits per pack.
    assert tuple(batch["input_ids"].shape) == (2, _SEQ)
    assert "example_ids" in batch  # packing marks example membership


def test_mixture_data_loader_buffered_packing(tmp_path):
    ds = [_FakeDataset(200, 10), _FakeDataset(100, 20)]
    coll = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    dl = MixtureDataLoader(
        ds,
        [0.5, 0.5],
        coll,
        work_dir=str(tmp_path),
        global_batch_size=2 * _SEQ,
        seed=0,
        pack=True,
        pack_max_crops=1,
        pack_buffer_size=4,
    )
    dl.reshuffle(epoch=1)
    assert dl.total_batches is None
    batch = next(iter(dl))
    assert tuple(batch["input_ids"].shape) == (2, _SEQ)
    assert "example_ids" in batch


def test_mixture_data_loader_buffered_packing_resumes_exactly(tmp_path):
    datasets = [_FakeDataset(200, 10), _FakeDataset(100, 20)]
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()

    def build_loader(work_dir, prefetch_workers):
        return MixtureDataLoader(
            datasets,
            [0.5, 0.5],
            collator,
            work_dir=work_dir,
            global_batch_size=2 * _SEQ,
            seed=17,
            pack=True,
            pack_max_crops=1,
            pack_buffer_size=4,
            prefetch_workers=prefetch_workers,
        )

    for original_workers, restored_workers in ((0, 0), (0, 4), (4, 0), (4, 4)):
        original = build_loader(tmp_path / f"original-{original_workers}", original_workers)
        original.reshuffle(epoch=3)
        original_iter = iter(original)
        next(original_iter)
        state = original.state_dict()
        expected = next(original_iter)
        original_iter.close()

        restored = build_loader(tmp_path / f"restored-{restored_workers}", restored_workers)
        restored.load_state_dict(state)
        restored.reshuffle()
        restored_iter = iter(restored)
        actual = next(restored_iter)
        restored_iter.close()

        for key in ("input_ids", "example_ids", "loss_masks", "position_ids"):
            np.testing.assert_array_equal(actual[key], expected[key])


def test_mixture_data_loader_buffered_resume_restores_only_bounded_state(tmp_path):
    datasets = [_CountingFakeDataset(200, 1000), _CountingFakeDataset(100, 2000)]
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()

    def build_loader(work_dir):
        return MixtureDataLoader(
            datasets,
            [0.5, 0.5],
            collator,
            work_dir=work_dir,
            global_batch_size=2 * _SEQ,
            seed=23,
            pack=True,
            pack_max_crops=1,
            pack_buffer_size=4,
            pack_image_weight=30.0,
            prefetch_workers=0,
        )

    original = build_loader(tmp_path / "original")
    original.reshuffle(epoch=2)
    original_iter = iter(original)
    for _ in range(10):
        next(original_iter)
    state = original.state_dict()
    expected = next(original_iter)
    original_iter.close()

    packing_state = state["packing_state"]
    assert packing_state["version"] == 5
    assert packing_state["pack_image_weight"] == 30.0
    assert packing_state["dataset_fingerprints"] == [None, None]
    assert packing_state["packs_emitted"] == 20
    assert len(packing_state["buffer_refs"]) <= 4

    for dataset in datasets:
        dataset.loads = 0
    restored = build_loader(tmp_path / "restored")
    restored.load_state_dict(state)
    restored.reshuffle()
    restored_iter = iter(restored)
    actual = next(restored_iter)
    restored_iter.close()
    fast_resume_loads = sum(dataset.loads for dataset in datasets)

    for key in ("input_ids", "example_ids", "loss_masks", "position_ids"):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert fast_resume_loads == len(packing_state["buffer_refs"]) + 2

    # Checkpoints written before cursor state was added remain resumable. They restart the
    # loader and replay the old prefix once; unlike state v5, this fallback cannot validate
    # source contents. Subsequent checkpoints use the bounded, fingerprinted path above.
    legacy_state = dict(state)
    legacy_state.pop("packing_state")
    for dataset in datasets:
        dataset.loads = 0
    legacy = build_loader(tmp_path / "legacy")
    legacy.load_state_dict(legacy_state)
    legacy.reshuffle()
    legacy_iter = iter(legacy)
    legacy_actual = next(legacy_iter)
    legacy_iter.close()

    np.testing.assert_array_equal(legacy_actual["input_ids"], expected["input_ids"])
    assert sum(dataset.loads for dataset in datasets) > fast_resume_loads


def test_mixture_data_loader_v5_validates_source_content_fingerprints(tmp_path):
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()

    def build_loader(work_dir, fingerprint):
        return MixtureDataLoader(
            [_FingerprintedFakeDataset(200, 1000, fingerprint), _FakeDataset(100, 2000)],
            [0.5, 0.5],
            collator,
            work_dir=work_dir,
            global_batch_size=2 * _SEQ,
            seed=29,
            pack=True,
            pack_max_crops=1,
            pack_buffer_size=4,
            dataset_names=["native-replay", "caption"],
        )

    original = build_loader(tmp_path / "original", "content-a")
    original.reshuffle(epoch=2)
    original_iter = iter(original)
    next(original_iter)
    state = original.state_dict()
    expected = next(original_iter)
    original_iter.close()

    fingerprints = state["packing_state"]["dataset_fingerprints"]
    assert fingerprints[0]["type"].endswith("._FingerprintedFakeDataset")
    assert fingerprints[0]["version"] == "fake-content-v1"
    assert fingerprints[0]["value"] == "content-a"
    assert fingerprints[1] is None

    restored = build_loader(tmp_path / "restored", "content-a")
    restored.load_state_dict(state)
    restored.reshuffle()
    restored_iter = iter(restored)
    actual = next(restored_iter)
    restored_iter.close()
    np.testing.assert_array_equal(actual["input_ids"], expected["input_ids"])

    changed = build_loader(tmp_path / "changed", "content-b")
    changed.load_state_dict(state)
    changed.reshuffle()
    with pytest.raises(
        OLMoConfigurationError,
        match="dataset content fingerprint changed for source 'native-replay'",
    ):
        next(iter(changed))


def test_mixture_data_loader_recognizes_native_text_replay_fingerprint():
    dataset = object.__new__(NativeTextReplayDataset)
    setattr(dataset, "manifest", SimpleNamespace(content_fingerprint="a" * 64))

    fingerprint = MixtureDataLoader._dataset_fingerprint(dataset, "native-replay")

    assert fingerprint == {
        "type": "olmo_core.data.multimodal.native_text_replay.NativeTextReplayDataset",
        "version": "native-text-replay-v2",
        "value": "a" * 64,
    }


def test_mixture_data_loader_can_require_fingerprinted_resume_for_legacy_v4(tmp_path):
    datasets = [_FingerprintedFakeDataset(200, 1000, "content-a")]
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()

    def build_loader(work_dir, *, allow_legacy=True):
        return MixtureDataLoader(
            datasets,
            [1.0],
            collator,
            work_dir=work_dir,
            global_batch_size=2 * _SEQ,
            seed=43,
            pack=True,
            pack_max_crops=1,
            pack_buffer_size=4,
            dataset_names=["native-replay"],
            allow_legacy_state_without_dataset_fingerprints=allow_legacy,
        )

    original = build_loader(tmp_path / "original")
    original.reshuffle(epoch=1)
    original_iter = iter(original)
    next(original_iter)
    state = original.state_dict()
    expected = next(original_iter)
    original_iter.close()

    packing_state = dict(state["packing_state"])
    packing_state["version"] = 4
    packing_state.pop("dataset_fingerprints")
    legacy_state = dict(state, packing_state=packing_state)

    rejected = build_loader(tmp_path / "rejected", allow_legacy=False)
    rejected.load_state_dict(legacy_state)
    rejected.reshuffle()
    with pytest.raises(
        OLMoConfigurationError,
        match="allow_legacy_state_without_dataset_fingerprints=True",
    ):
        next(iter(rejected))

    opted_in = build_loader(tmp_path / "opted-in", allow_legacy=True)
    opted_in.load_state_dict(legacy_state)
    opted_in.reshuffle()
    opted_in_iter = iter(opted_in)
    actual = next(opted_in_iter)
    opted_in_iter.close()
    np.testing.assert_array_equal(actual["input_ids"], expected["input_ids"])


def test_mixture_data_loader_prefetch_skips_errors_in_reference_order(tmp_path):
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()

    def build_loader(work_dir, workers):
        dataset = _FailingFakeDataset(200, 1000)
        loader = MixtureDataLoader(
            [dataset],
            [1.0],
            collator,
            work_dir=work_dir,
            global_batch_size=2 * _SEQ,
            seed=31,
            pack=True,
            pack_max_crops=1,
            pack_buffer_size=4,
            prefetch_workers=workers,
        )
        loader.reshuffle(epoch=2)
        refs = list(itertools.islice(loader._rank_refs_from_cursor(), 3))
        dataset.fail_indices = {refs[0][1], refs[2][1]}
        return loader

    sync = build_loader(tmp_path / "sync", 0)
    sync_iter = iter(sync)
    expected = next(sync_iter)
    sync_state = sync.state_dict()
    sync_iter.close()

    threaded = build_loader(tmp_path / "threaded", 4)
    threaded_iter = iter(threaded)
    actual = next(threaded_iter)
    threaded_state = threaded.state_dict()
    threaded_iter.close()

    np.testing.assert_array_equal(actual["input_ids"], expected["input_ids"])
    assert sync_state["total_data_errors"] == threaded_state["total_data_errors"] == 2
    assert (
        sync_state["packing_state"]["refs_consumed"]
        == threaded_state["packing_state"]["refs_consumed"]
    )


def test_mixture_data_loader_nonpacked_prefetch_preserves_order_and_errors(tmp_path):
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()

    def build_loader(work_dir, workers):
        dataset = _FailingFakeDataset(200, 1000)
        loader = MixtureDataLoader(
            [dataset],
            [1.0],
            collator,
            work_dir=work_dir,
            global_batch_size=2 * _SEQ,
            seed=47,
            prefetch_workers=workers,
            max_consecutive_data_errors=0,
            max_total_data_errors=0,
            dataset_names=["audited-test"],
        )
        loader.reshuffle(epoch=3)
        assert loader._order is not None
        failed_ref = loader._order[1]
        dataset.fail_indices = {failed_ref[1]}
        loader.allowed_data_error_signatures = {
            ("audited-test", failed_ref[1], failed_ref[2]): (
                ValueError,
                f"synthetic failure for {failed_ref[1]}",
            )
        }
        return loader

    sync = build_loader(tmp_path / "sync-nonpacked", 0)
    sync_iter = iter(sync)
    expected = [next(sync_iter), next(sync_iter)]
    sync_state = sync.state_dict()
    expected_after_resume = next(sync_iter)
    sync_iter.close()

    threaded = build_loader(tmp_path / "threaded-nonpacked", 4)
    threaded_iter = iter(threaded)
    actual = [next(threaded_iter), next(threaded_iter)]
    threaded_state = threaded.state_dict()
    threaded_iter.close()

    restored = build_loader(tmp_path / "restored-nonpacked", 4)
    restored.load_state_dict(threaded_state)
    restored.reshuffle()
    restored_iter = iter(restored)
    actual_after_resume = next(restored_iter)
    restored_iter.close()

    for actual_batch, expected_batch in zip(actual, expected):
        for key in ("input_ids", "loss_masks", "position_ids"):
            np.testing.assert_array_equal(actual_batch[key], expected_batch[key])
    assert sync_state["batches_processed"] == threaded_state["batches_processed"] == 2
    assert sync_state["total_data_errors"] == threaded_state["total_data_errors"] == 1
    for key in ("input_ids", "loss_masks", "position_ids"):
        np.testing.assert_array_equal(actual_after_resume[key], expected_after_resume[key])


def test_mixture_data_loader_allowlist_requires_exact_error_signature(tmp_path, caplog):
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    loader = MixtureDataLoader(
        [_FakeDataset(10, 1)],
        [1.0],
        collator,
        work_dir=tmp_path,
        global_batch_size=2 * _SEQ,
        max_consecutive_data_errors=0,
        max_total_data_errors=0,
        dataset_names=["audited-test"],
        allowed_data_error_signatures={("audited-test", 7, 0): (ValueError, "known malformed row")},
    )

    loader._handle_data_error((0, 7, 0), ValueError("known malformed row"))
    assert loader.total_data_errors == 1
    assert "explicitly allowlisted data error" in caplog.text

    with pytest.raises(ValueError, match="different failure"):
        loader._handle_data_error((0, 7, 0), ValueError("different failure"))
    assert loader.total_data_errors == 2


@pytest.mark.parametrize("restored_workers", [0, 4])
def test_mixture_data_loader_resume_preserves_tolerated_errors(tmp_path, restored_workers):
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    dataset = _FailingFakeDataset(200, 1000)

    def build_loader(work_dir, workers):
        return MixtureDataLoader(
            [dataset],
            [1.0],
            collator,
            work_dir=work_dir,
            global_batch_size=2 * _SEQ,
            seed=41,
            pack=True,
            pack_max_crops=1,
            pack_buffer_size=4,
            prefetch_workers=workers,
            dataset_names=["academic-test"],
        )

    original = build_loader(tmp_path / "original", 4)
    original.reshuffle(epoch=1)
    refs = list(itertools.islice(original._rank_refs_from_cursor(), 3))
    dataset.fail_indices = {refs[0][1], refs[2][1]}
    original_iter = iter(original)
    next(original_iter)
    state = original.state_dict()
    expected = next(original_iter)
    expected_state = original.state_dict()
    original_iter.close()

    restored = build_loader(tmp_path / f"restored-{restored_workers}", restored_workers)
    restored.load_state_dict(state)
    restored.reshuffle()
    restored_iter = iter(restored)
    actual = next(restored_iter)
    actual_state = restored.state_dict()
    restored_iter.close()

    for key in ("input_ids", "example_ids", "loss_masks", "position_ids"):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert state["total_data_errors"] == 2
    assert actual_state["total_data_errors"] == expected_state["total_data_errors"]
    assert (
        actual_state["packing_state"]["refs_consumed"]
        == expected_state["packing_state"]["refs_consumed"]
    )


@pytest.mark.parametrize(
    ("max_consecutive", "max_total"),
    [(2, 10), (10, 2)],
)
def test_mixture_data_loader_error_limits_allow_n_and_fail_on_n_plus_one(
    tmp_path, max_consecutive, max_total
):
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    loader = MixtureDataLoader(
        [_FakeDataset(10, 1)],
        [1.0],
        collator,
        work_dir=tmp_path,
        global_batch_size=2 * _SEQ,
        max_consecutive_data_errors=max_consecutive,
        max_total_data_errors=max_total,
        dataset_names=["academic-test"],
    )

    limit = min(max_consecutive, max_total)
    for index in range(limit):
        loader._handle_data_error((0, index, 0), ValueError(f"failure {index}"))
    with pytest.raises(ValueError, match=f"failure {limit}"):
        loader._handle_data_error((0, limit, 0), ValueError(f"failure {limit}"))

    assert loader.total_data_errors == limit + 1


def test_mixture_data_loader_error_limit_reports_exact_reference(tmp_path, caplog):
    dataset = _FailingFakeDataset(200, 1000)
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    loader = MixtureDataLoader(
        [dataset],
        [1.0],
        collator,
        work_dir=tmp_path,
        global_batch_size=2 * _SEQ,
        seed=37,
        pack=True,
        pack_max_crops=1,
        pack_buffer_size=4,
        max_consecutive_data_errors=0,
        max_total_data_errors=0,
        dataset_names=["academic-test"],
    )
    loader.reshuffle(epoch=1)
    first_ref = next(loader._rank_refs_from_cursor())
    dataset.fail_indices = {first_ref[1]}

    with pytest.raises(ValueError, match="synthetic failure") as exc_info:
        next(iter(loader))

    context = f"academic-test[{first_ref[1]}] at source epoch {first_ref[2]}"
    assert any(context in note for note in exc_info.value.__notes__)
    assert context in caplog.text
    assert loader.total_data_errors == 1
    assert loader.state_dict()["total_data_errors"] == 1


def test_mixture_data_loader_normalizes_weights(tmp_path):
    ds = [_FakeDataset(10, 1), _FakeDataset(10, 2)]
    coll = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    dl = MixtureDataLoader(
        ds, [3.0, 1.0], coll, work_dir=str(tmp_path), global_batch_size=2 * _SEQ, seed=0
    )
    np.testing.assert_allclose(dl.weights, [0.75, 0.25])


@pytest.mark.parametrize("weights", [[1.0, 0.0], [1.0, float("nan")], [1.0, float("inf")]])
def test_mixture_data_loader_rejects_nonpositive_or_nonfinite_weights(tmp_path, weights):
    datasets = [_FakeDataset(10, 1), _FakeDataset(10, 2)]
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()

    with pytest.raises(OLMoConfigurationError, match="finite and strictly positive"):
        MixtureDataLoader(
            datasets,
            weights,
            collator,
            work_dir=str(tmp_path),
            global_batch_size=2 * _SEQ,
            seed=0,
        )


def test_buffered_mixture_reference_stream_matches_molmo2(tmp_path):
    sizes = [7, 5, 11]
    weights = [0.6, 0.3, 0.1]
    seed = 95818
    world_size = 4
    n_per_rank = 80

    def molmo2_refs():
        rates = np.asarray(weights, dtype=np.float64)
        rates = np.asarray(rates / rates.sum(), dtype=np.float32)
        rng = np.random.RandomState(seed)
        counts = np.zeros(len(sizes), dtype=np.int64)
        shuffled = [(None, None) for _ in sizes]
        while True:
            source = int(rng.choice(len(sizes), p=rates))
            source_count = int(counts[source])
            counts[source] += 1
            source_epoch = source_count // sizes[source]
            shuffled_for, order = shuffled[source]
            if shuffled_for != source_epoch:
                order = np.arange(sizes[source], dtype=np.int32)
                make_random_state(seed, source_epoch, 1).shuffle(order)
                shuffled[source] = (source_epoch, order)
            yield source, int(order[source_count % sizes[source]]), source_epoch

    expected = list(itertools.islice(molmo2_refs(), world_size * n_per_rank))
    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()
    actual_by_rank = []
    for rank in range(world_size):
        loader = MixtureDataLoader(
            [_FakeDataset(size, 10 + i) for i, size in enumerate(sizes)],
            weights,
            collator,
            work_dir=tmp_path / str(rank),
            global_batch_size=world_size * _SEQ,
            seed=seed,
            pack=True,
            pack_max_crops=1,
            pack_buffer_size=4,
            dp_world_size=world_size,
            dp_rank=rank,
        )
        actual_by_rank.append(list(itertools.islice(loader._rank_refs_from_cursor(), n_per_rank)))
        resumed = list(itertools.islice(loader._rank_refs_from_cursor(17), 10))
        assert resumed == expected[rank + 17 * world_size :: world_size][:10]

    actual = [actual_by_rank[rank][i] for i in range(n_per_rank) for rank in range(world_size)]
    assert actual == expected


# ---------------------------------------------------------------------------
# PixMoCap style_and_length_v2 conditioning (Gap 1 vs mm_olmo)
# ---------------------------------------------------------------------------


class _FakeTok:
    """Minimal tokenizer for CPU tests: records the prompts it templates."""

    eos_token_id = 1
    bos_token_id = 0

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        prompt = messages[0]["content"]
        if prompt:
            self.prompts.append(prompt)
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text, add_special_tokens=False):
        if not text.startswith(" "):
            self.prompts.append(text)
        return [(ord(c) % 90) + 10 for c in text]


def _pixmo_cap(mode, **kw):
    from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig

    kw.setdefault("message_format", "document")
    cfg = PixMoCapDatasetConfig(
        dataset_path="synthetic", mode=mode, max_sequence_length=4096, seed=0, **kw
    )
    return cfg.build(_FakeTok())


def test_pixmo_cap_can_fail_closed_when_named_split_is_required():
    with pytest.raises(ValueError, match="does not provide named splits"):
        _pixmo_cap("caption", split="validation", require_split=True)


def test_pixmo_cap_style_length_prefix_format():
    ds = _pixmo_cap("caption")
    rng = np.random.RandomState(0)
    text = "x" * 300  # 300 chars -> bucket ~ 300//15 = 20
    with_num = 0
    for _ in range(400):
        p = ds._style_length_prefix("long_caption", text, rng)
        assert p.startswith("long_caption") and p.endswith(":")
        rest = p[len("long_caption") : -1]
        if rest:  # " <n>"
            with_num += 1
            assert -10 <= int(rest) <= 50  # ~20 +/- noise/15
    assert 0.80 < with_num / 400 < 0.97  # ~90% include the length bucket


def test_pixmo_cap_select_branches_styles():
    from olmo_core.data.multimodal.pixmo_cap import CAPTION_STYLE, TRANSCRIPT_STYLE

    row = {"caption": "a cat", "transcripts": ["spoken one", "spoken two"]}
    rng = np.random.RandomState(0)
    assert [s for s, _ in _pixmo_cap("caption")._select_branches(row, rng)] == [CAPTION_STYLE]
    assert [s for s, _ in _pixmo_cap("transcript")._select_branches(row, rng)] == [TRANSCRIPT_STYLE]
    both = _pixmo_cap("transcript_and_caption")._select_branches(row, rng)
    assert [s for s, _ in both] == [CAPTION_STYLE, TRANSCRIPT_STYLE]


def test_pixmo_cap_transcript_fallback_is_backward_compatible_and_can_be_disabled():
    from olmo_core.data.multimodal.pixmo_cap import CAPTION_STYLE

    row = {"caption": "caption fallback", "transcripts": []}
    rng = np.random.RandomState(0)

    assert _pixmo_cap("transcript")._select_branches(row, rng) == [
        (CAPTION_STYLE, "caption fallback")
    ]
    strict = _pixmo_cap("transcript", require_transcript=True)
    with pytest.raises(ValueError, match="requires at least one non-blank transcript"):
        strict._select_branches(row, rng)
    with pytest.raises(ValueError, match="requires at least one non-blank transcript"):
        strict._select_branches({"caption": "caption", "transcripts": ["", "  "]}, rng)


def test_pixmo_cap_validates_strict_transcript_completeness_without_loading_images(tmp_path):
    jsonl_path = tmp_path / "pixmo-cap.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image": "missing-image-a.png",
                        "caption": "caption a",
                        "transcripts": ["spoken a"],
                    }
                ),
                json.dumps(
                    {
                        "image": "missing-image-b.png",
                        "caption": "caption b",
                        "transcripts": ["", "  "],
                    }
                ),
                json.dumps({"image": "missing-image-c.png", "caption": "caption c"}),
            ]
        )
        + "\n"
    )
    from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig

    dataset = PixMoCapDatasetConfig(
        dataset_path=str(jsonl_path),
        mode="transcript",
        require_transcript=True,
        message_format="document",
    ).build(_FakeTok())

    with pytest.raises(ValueError, match=r"found 2 invalid rows out of 3.*\[1, 2\]"):
        dataset.validate_required_annotations()


def test_pixmo_cap_validates_arrow_and_synthetic_transcripts_without_image_access():
    class TranscriptOnlyArrow:
        def __init__(self):
            self.transcript_column_reads = 0

        def __getitem__(self, column):
            assert column == "transcripts"
            self.transcript_column_reads += 1
            return [["spoken a"], ["spoken b"]]

        def __len__(self):
            return 2

    arrow = TranscriptOnlyArrow()
    dataset = _pixmo_cap("transcript", require_transcript=True)
    dataset._kind = "arrow"
    dataset._hf = arrow
    dataset.validate_required_annotations()
    assert arrow.transcript_column_reads == 1

    _pixmo_cap("transcript", require_transcript=True).validate_required_annotations()


def test_pixmo_cap_conditioning_injects_per_branch_prefix():
    ds = _pixmo_cap("transcript_and_caption", style_length_conditioning=True)
    seq = ds[0]
    # two branches -> subsegment ids present, two distinct annotations
    assert "subsegment_ids" in seq
    assert len(set(seq["subsegment_ids"].tolist())) == 3  # prefix + 2 branches
    # the two user turns were templated with the long_caption / transcript style prefixes
    prompts = ds.tokenizer.prompts
    assert any(p.startswith("long_caption") and ":" in p for p in prompts)
    assert any(p.startswith("transcript") and ":" in p for p in prompts)


def test_pixmo_cap_fixed_prompt_disables_conditioning():
    ds = _pixmo_cap("caption", fixed_prompt="Describe this image.")
    _ = ds[0]
    assert ds.tokenizer.prompts == ["Describe this image."]  # verbatim, no style prefix


def test_pixmo_cap_conditioning_off():
    ds = _pixmo_cap("caption", style_length_conditioning=False)
    _ = ds[0]
    # prompt is sampled from the pool verbatim, with no "long_caption ...:" prefix
    assert all(not p.startswith("long_caption") for p in ds.tokenizer.prompts)


def test_pointing_formats_messages_before_image_augmentation(monkeypatch):
    """Match Molmo2's shared-RNG order: dataset/formatter, shuffle, then image processor."""
    from olmo_core.data.multimodal import message_sequence
    from olmo_core.data.multimodal.pixmo_points import _build_example
    from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds

    events = []
    rng = np.random.RandomState(7)

    def build_branches(branch_rng):
        events.append(("format", int(branch_rng.randint(2**31))))
        return [("Locate it", "Here")]

    def fake_preprocess(*args, rng, **kwargs):
        events.append(("image", int(rng.randint(2**31))))
        return (
            torch.zeros(1, 1, 729, _PATCH_DIM),
            torch.zeros(1, 1, 4, dtype=torch.long),
            np.array([1, 1, 1, 1], dtype=np.int32),
        )

    monkeypatch.setattr(message_sequence, "preprocess_image_molmo2", fake_preprocess)
    _build_example(
        _FakeTok(),
        None,
        build_branches,
        max_crops=8,
        loss_token_weighting="none",
        token_ids=Molmo2TokenIds(),
        rng=rng,
    )

    reference_rng = np.random.RandomState(7)
    assert events == [
        ("format", int(reference_rng.randint(2**31))),
        ("image", int(reference_rng.randint(2**31))),
    ]


# ---------------------------------------------------------------------------
# Truncated/corrupt image tolerance (multi-node run robustness)
# ---------------------------------------------------------------------------


def test_truncated_image_preprocesses_without_raising():
    """A truncated PixMo image must not raise (PIL OSError) — it would crash a data-worker
    thread and, under distributed packing, hang the other ranks into a NCCL watchdog abort."""
    import io

    import torch
    from PIL import Image

    from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2

    buf = io.BytesIO()
    Image.fromarray((np.random.rand(64, 96, 3) * 255).astype("uint8")).save(buf, format="JPEG")
    truncated = Image.open(io.BytesIO(buf.getvalue()[:-200]))  # drop trailing bytes
    crops, pooled, grid = preprocess_image_molmo2(
        truncated, dtype=torch.float32, device=torch.device("cpu"), max_crops=8
    )
    assert crops.shape[0] == 1 and grid.shape == (4,)
