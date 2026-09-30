"""Exact-output and resume tests for opt-in direct packed-image collation."""

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from olmo_core.config import Config
from olmo_core.data.multimodal.collator import MultimodalCollator
from olmo_core.data.multimodal.mixture_data_loader import (
    MixtureDataLoader,
    MixtureDataLoaderConfig,
)
from olmo_core.data.multimodal.packing import (
    _PackedImageParts,
    iter_packs,
    pack_examples,
)
from olmo_core.exceptions import OLMoConfigurationError


def _example(tag, crops, size=7, dtype=np.float32):
    tokens = np.arange(tag, tag + size, dtype=np.int64)
    images = np.arange(crops * 4 * 6, dtype=np.float32).reshape(crops, 4, 6)
    images = (images / 7.0 + tag).astype(dtype)
    pooled = np.arange(crops * 2, dtype=np.int64).reshape(crops, 2)
    pooled[:, 1] = -1
    return {
        "input_ids": tokens,
        "labels": tokens + 1,
        "loss_masks": np.asarray([0.0] + [0.5] * (size - 1), dtype=np.float32),
        "position_ids": np.arange(size, dtype=np.int64),
        "token_type_ids": np.asarray([1] * crops + [0] * (size - crops), dtype=np.int64),
        "subsegment_ids": np.arange(size, dtype=np.int64) // 3,
        "images": images,
        "pooled_patches_idx": pooled,
        "_source_name": f"source-{tag}",
    }


def _assert_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(
            actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
        )
    elif isinstance(actual, np.ndarray):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("crops", [(2, 0, 1), (0, 0), (1, 3)])
@pytest.mark.parametrize("sequence_length", [8, 32])
def test_deferred_images_preserve_every_collated_field_and_truncation(crops, sequence_length):
    examples = [_example(10 + index, count) for index, count in enumerate(crops)]
    before = copy.deepcopy(examples)
    packed = pack_examples(examples)
    deferred = pack_examples(examples, defer_image_copy=True)
    assert isinstance(packed["images"], np.ndarray)
    for key in packed.keys() - {"images"}:
        _assert_equal(deferred[key], packed[key])
    assert deferred["images"].shape == packed["images"].shape
    if any(crops):
        assert isinstance(deferred["images"], _PackedImageParts)
        assert all(
            part is example["images"]
            for part, example in zip(
                deferred["images"].parts,
                (example for example in examples if example["images"].shape[0]),
            )
        )
    collator = MultimodalCollator(pad_token_id=0, pad_sequence_length=sequence_length)
    text_only = pack_examples([_example(99, 0)])
    _assert_equal(collator([deferred, text_only]), collator([packed, text_only]))
    _assert_equal(examples, before)
    if any(crops):
        assert np.array_equal(
            packed["pooled_patches_idx"][:, 1], -np.ones(sum(crops), dtype=np.int64)
        )
        expected_first = np.concatenate(
            [np.arange(count) * 2 + sum(crops[:index]) * 4 for index, count in enumerate(crops)]
        )
        assert np.array_equal(packed["pooled_patches_idx"][:, 0], expected_first)
    else:
        assert collator([deferred])["images"].shape[1] == 1
        assert not torch.count_nonzero(collator([deferred])["images"])


def test_deferred_images_keep_noncontiguous_values_and_float32_bits():
    examples = [_example(10, 2), _example(20, 1)]
    for example in examples:
        example["images"] = example["images"][:, :, ::-1]
    examples[0]["images"][0, 0, :4] = np.array(
        [0x80000000, 0x7FC00001, 0x7F800000, 0xFF800000], dtype=np.uint32
    ).view(np.float32)
    collator = MultimodalCollator(pad_token_id=0, pad_sequence_length=32)
    baseline = collator([pack_examples(examples)])
    candidate = collator([pack_examples(examples, defer_image_copy=True)])
    _assert_equal(candidate, baseline)
    examples[0]["images"][:] = 42
    _assert_equal(candidate, baseline)


def test_deferred_images_skip_intermediate_image_concatenation(monkeypatch):
    examples = [_example(10, 2), _example(20, 1)]
    native_concatenate = np.concatenate
    image_copies = []

    def record(arrays, *args, **kwargs):
        if arrays and arrays[0].ndim == 3:
            image_copies.append(sum(array.nbytes for array in arrays))
        return native_concatenate(arrays, *args, **kwargs)

    monkeypatch.setattr(np, "concatenate", record)
    collator = MultimodalCollator(pad_token_id=0, pad_sequence_length=32)
    baseline = collator([pack_examples(examples)])
    assert image_copies == [sum(example["images"].nbytes for example in examples)]
    image_copies.clear()
    candidate = collator([pack_examples(examples, defer_image_copy=True)])
    assert not image_copies
    _assert_equal(candidate, baseline)


@pytest.mark.parametrize(
    "dtypes",
    [
        (np.float16, np.float32),
        (np.float64, np.float32),
        (np.int64, np.float64),
        (">f4", np.float32),
    ],
)
def test_non_native_float32_inputs_keep_legacy_dtype_promotion(dtypes):
    examples = [_example(10 + index, 2, dtype=dtype) for index, dtype in enumerate(dtypes)]
    baseline = pack_examples(examples)
    candidate = pack_examples(examples, defer_image_copy=True)
    assert isinstance(candidate["images"], np.ndarray)
    _assert_equal(candidate, baseline)
    collator = MultimodalCollator(pad_token_id=0, pad_sequence_length=32)
    _assert_equal(collator([candidate]), collator([baseline]))


@pytest.mark.parametrize("buffer_size", [0, 4])
def test_deferred_finite_packing_keeps_selection_flush_and_oversize_behavior(buffer_size):
    examples = [_example(index + 10, index % 3, size=7 + index % 4) for index in range(12)]
    examples.insert(4, _example(70, 5, size=40))
    kwargs = {"seq_len": 32, "max_crops_per_pack": 4, "buffer_size": buffer_size}
    baseline = list(iter_packs(examples, **kwargs))
    candidate = list(iter_packs(examples, defer_image_copy=True, **kwargs))
    assert len(candidate) == len(baseline)
    collator = MultimodalCollator(pad_token_id=0, pad_sequence_length=32)
    _assert_equal(collator(candidate), collator(baseline))


class _Dataset:
    def __init__(self, tag):
        self.tag = tag
        self.content_fingerprint = f"packed-image-copy-{tag}"

    def __len__(self):
        return 31

    def get(self, index, epoch):
        return _example(self.tag + index + epoch * 100, index % 3, size=5 + index % 7)


def _loader(path, deferred, grouped, *, rank=0, pack=True, buffer_size=4):
    return MixtureDataLoader(
        [_Dataset(100), _Dataset(200)],
        [0.4, 0.6],
        MultimodalCollator(pad_token_id=0, pad_sequence_length=64),
        work_dir=path,
        global_batch_size=8 * 64,
        seed=31,
        epoch_instances=2400,
        pack=pack,
        pack_max_crops=8,
        pack_buffer_size=buffer_size,
        pack_image_weight=30.0,
        prefetch_workers=2,
        defer_packed_image_copy=deferred,
        dp_world_size=2,
        dp_rank=rank,
        dataset_names=["caption", "transcript"],
        source_groups={"caption": "image", "transcript": "text"} if grouped else None,
        group_sequence_quotas={"image": 4, "text": 4} if grouped else None,
    )


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("pack,buffer_size", [(False, 0), (True, 0), (True, 4)])
@pytest.mark.parametrize("rank", [0, 1])
def test_loader_direct_image_copy_preserves_exact_batches_state_and_mock(
    tmp_path, grouped, pack, buffer_size, rank
):
    loaders = [
        _loader(tmp_path, deferred, grouped, rank=rank, pack=pack, buffer_size=buffer_size)
        for deferred in (False, True)
    ]
    _assert_equal(loaders[0].get_mock_batch(), loaders[1].get_mock_batch())
    for child in loaders[1]._group_loaders.values():
        assert child.defer_packed_image_copy is True
    iterators = []
    try:
        for loader in loaders:
            loader.reshuffle(epoch=2)
            iterators.append(iter(loader))
        for _ in range(5):
            _assert_equal(next(iterators[0]), next(iterators[1]))
            _assert_equal(loaders[0].state_dict(), loaders[1].state_dict())
    finally:
        for iterator in iterators:
            iterator.close()


@pytest.mark.parametrize("before,after", [(False, True), (True, False)])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("buffer_size", [0, 4])
def test_loader_direct_image_copy_can_change_on_exact_resume(
    tmp_path, before, after, grouped, buffer_size
):
    original = _loader(tmp_path, before, grouped, rank=1, buffer_size=buffer_size)
    original.reshuffle(epoch=3)
    iterator = iter(original)
    try:
        for _ in range(3):
            next(iterator)
        state = copy.deepcopy(original.state_dict())
        expected = [next(iterator) for _ in range(4)]
        expected_state = copy.deepcopy(original.state_dict())
    finally:
        iterator.close()
    restored = _loader(tmp_path, after, grouped, rank=1, buffer_size=buffer_size)
    restored.load_state_dict(state)
    restored.reshuffle()
    iterator = iter(restored)
    try:
        _assert_equal([next(iterator) for _ in expected], expected)
        _assert_equal(restored.state_dict(), expected_state)
    finally:
        iterator.close()


@dataclass
class _ContainerConfig(Config):
    data_loader: MixtureDataLoaderConfig


@pytest.mark.parametrize("flag", [None, False, True])
def test_direct_image_copy_config_build_roundtrip_and_nested_default_omission(tmp_path, flag):
    config = MixtureDataLoaderConfig(
        global_batch_size=128,
        sequence_length=16,
        work_dir=str(tmp_path),
        defer_packed_image_copy=flag,
    )
    saved = _ContainerConfig(config).as_config_dict()["data_loader"]
    if flag is None:
        assert "defer_packed_image_copy" not in saved
    else:
        assert saved["defer_packed_image_copy"] is flag
    restored = MixtureDataLoaderConfig.from_dict(saved)
    assert restored == config
    dataset: Any = SimpleNamespace(
        datasets=[_Dataset(100)],
        weights=[1.0],
        names=["caption"],
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    assert restored.build(dataset).defer_packed_image_copy is bool(flag)


@pytest.mark.parametrize("flag", [0, 1, "true"])
def test_direct_image_copy_rejects_non_boolean_flags(tmp_path, flag):
    with pytest.raises(OLMoConfigurationError, match="defer_packed_image_copy"):
        _loader(tmp_path, flag, False)
