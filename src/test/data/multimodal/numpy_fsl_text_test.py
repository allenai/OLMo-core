"""Tests for adapting fixed-length numpy text data to multimodal examples."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from olmo_core.data import (
    InstanceFilterConfig,
    NumpyFSLDataset,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.multimodal import (
    MixtureDataLoader,
    MultimodalCollatorConfig,
    NumpyFSLTextDataset,
    NumpyFSLTextDatasetConfig,
)
from olmo_core.data.numpy_dataset import NumpyFSLDatasetMixture
from olmo_core.nn.vision.molmo2_tokens import N_PATCHES_SQ, PATCH_DIM, POOL_H, POOL_W


def _write_mmap(path: Path, values, dtype) -> None:
    mmap = np.memmap(path, mode="w+", dtype=dtype, shape=(len(values),))
    mmap[:] = values
    mmap.flush()


def test_numpy_fsl_text_dataset_shifts_labels_and_masks(tmp_path: Path):
    token_path = tmp_path / "tokens.npy"
    mask_path = tmp_path / "mask.npy"
    _write_mmap(token_path, [10, 11, 12, 13], np.uint16)
    _write_mmap(mask_path, [True, False, True, False], np.bool_)
    child = NumpyFSLDataset(
        token_path,
        sequence_length=4,
        pad_token_id=0,
        eos_token_id=1,
        vocab_size=32_000,
        label_mask_paths=[mask_path],
        metadata={"source": "unit-test"},
    )

    example = NumpyFSLTextDataset(child)[0]

    np.testing.assert_array_equal(example["input_ids"], [10, 11, 12, 13])
    np.testing.assert_array_equal(example["labels"], [-100, 12, -100, -100])
    np.testing.assert_array_equal(example["loss_masks"], [0.0, 1.0, 0.0, 0.0])
    assert example["loss_masks"].sum() == np.count_nonzero(example["labels"] != -100)
    np.testing.assert_array_equal(example["position_ids"], [0, 1, 2, 3])
    np.testing.assert_array_equal(example["token_type_ids"], [0, 0, 0, 0])
    assert example["metadata"] == {"source": "unit-test"}


def test_numpy_fsl_text_dataset_preserves_repetition_filter_divisor(tmp_path: Path):
    token_path = tmp_path / "tokens.npy"
    mask_path = tmp_path / "mask.npy"
    _write_mmap(token_path, [7, 7, 7, 7], np.uint16)
    _write_mmap(mask_path, [True, False, False, False], np.bool_)
    child = NumpyFSLDataset(
        token_path,
        sequence_length=4,
        pad_token_id=0,
        eos_token_id=1,
        vocab_size=32_000,
        instance_filter_config=InstanceFilterConfig(
            repetition_min_period=1,
            repetition_max_period=1,
            repetition_max_count=2,
        ),
        label_mask_paths=[mask_path],
    )
    assert child[0]["instance_mask"] is False

    example = NumpyFSLTextDataset(child)[0]

    np.testing.assert_array_equal(example["labels"], [-100, -100, -100, -100])
    # Match OLMoDDP: a filtered row contributes L-1 divisor weight regardless of label_mask.
    np.testing.assert_array_equal(example["loss_masks"], [1.0, 1.0, 1.0, 0.0])


def test_numpy_fsl_text_dataset_emits_empty_vision_schema(tmp_path: Path):
    token_path = tmp_path / "tokens.npy"
    _write_mmap(token_path, [10, 11, 12, 13], np.uint16)
    dataset = NumpyFSLTextDataset(
        NumpyFSLDataset(
            token_path,
            sequence_length=4,
            pad_token_id=0,
            eos_token_id=1,
            vocab_size=32_000,
        )
    )

    example = dataset[0]
    assert example["images"].shape == (0, N_PATCHES_SQ, PATCH_DIM)
    assert example["images"].dtype == np.float32
    assert example["pooled_patches_idx"].shape == (0, POOL_H * POOL_W)
    assert example["pooled_patches_idx"].dtype == np.int64

    collator = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=4).build()
    loader = MixtureDataLoader(
        [dataset],
        [1.0],
        collator,
        work_dir=tmp_path,
        global_batch_size=4,
        epoch_instances=1,
    )
    loader.reshuffle(epoch=1)
    batch = next(iter(loader))
    assert tuple(batch["images"].shape) == (1, 1, N_PATCHES_SQ, PATCH_DIM)
    assert tuple(batch["pooled_patches_idx"].shape) == (1, 1, POOL_H * POOL_W)
    assert batch["image_crop_counts"].tolist() == [0]
    assert batch["pooled_token_counts"].tolist() == [0]
    assert (batch["pooled_patches_idx"] == -1).all()


def test_numpy_fsl_text_dataset_fingerprint_combines_adapter_and_child():
    child = Mock(
        sequence_length=4,
        fingerprint="a" * 64,
        fingerprint_version="child-v1",
        instance_filter_config=None,
        label_mask_paths=None,
        generate_doc_lengths=False,
    )
    dataset = NumpyFSLTextDataset(child)

    assert dataset.content_fingerprint_version == "numpy-fsl-text-adapter-v1"
    assert dataset.fingerprint_version == dataset.content_fingerprint_version
    assert dataset.fingerprint == dataset.content_fingerprint
    assert len(dataset.content_fingerprint) == 64
    assert dataset.content_fingerprint == NumpyFSLTextDataset(child).content_fingerprint

    changed_child = Mock(
        sequence_length=4,
        fingerprint="b" * 64,
        fingerprint_version="child-v1",
        instance_filter_config=None,
        label_mask_paths=None,
        generate_doc_lengths=False,
    )
    assert NumpyFSLTextDataset(changed_child).content_fingerprint != dataset.content_fingerprint
    changed_child_version = Mock(
        sequence_length=4,
        fingerprint=child.fingerprint,
        fingerprint_version="child-v2",
        instance_filter_config=None,
        label_mask_paths=None,
        generate_doc_lengths=False,
    )
    assert (
        NumpyFSLTextDataset(changed_child_version).content_fingerprint
        != dataset.content_fingerprint
    )
    changed_length = Mock(
        sequence_length=8,
        fingerprint=child.fingerprint,
        fingerprint_version=child.fingerprint_version,
        instance_filter_config=None,
        label_mask_paths=None,
        generate_doc_lengths=False,
    )
    assert NumpyFSLTextDataset(changed_length).content_fingerprint != dataset.content_fingerprint


def test_numpy_fsl_text_fingerprint_includes_filter_and_label_masks(tmp_path: Path):
    token_path = tmp_path / "tokens.npy"
    first_mask_path = tmp_path / "first-mask.npy"
    second_mask_path = tmp_path / "second-mask.npy"
    _write_mmap(token_path, [10, 11, 12, 13], np.uint16)
    _write_mmap(first_mask_path, [True, True, True, True], np.bool_)
    _write_mmap(second_mask_path, [True, False, True, False], np.bool_)

    def build(*, repetition_max_count: int, mask_path: Path) -> NumpyFSLTextDataset:
        return NumpyFSLTextDataset(
            NumpyFSLDataset(
                token_path,
                sequence_length=4,
                pad_token_id=0,
                eos_token_id=1,
                vocab_size=32_000,
                instance_filter_config=InstanceFilterConfig(
                    repetition_min_period=1,
                    repetition_max_period=1,
                    repetition_max_count=repetition_max_count,
                ),
                label_mask_paths=[mask_path],
            )
        )

    baseline = build(repetition_max_count=2, mask_path=first_mask_path)
    changed_filter = build(repetition_max_count=3, mask_path=first_mask_path)
    changed_mask = build(repetition_max_count=2, mask_path=second_mask_path)

    assert baseline.dataset.fingerprint == changed_filter.dataset.fingerprint
    assert baseline.dataset.fingerprint == changed_mask.dataset.fingerprint
    assert baseline.content_fingerprint != changed_filter.content_fingerprint
    assert baseline.content_fingerprint != changed_mask.content_fingerprint


def test_numpy_fsl_text_dataset_config_and_prepare_delegate(tmp_path: Path):
    token_path = tmp_path / "tokens.npy"
    _write_mmap(token_path, [10, 11, 12, 13], np.uint16)
    config = NumpyFSLTextDatasetConfig(
        dataset=NumpyFSLDatasetConfig(
            paths=[str(token_path)],
            sequence_length=4,
            tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=1, pad_token_id=0),
        )
    )

    dataset = config.build()
    with patch.object(dataset.dataset, "prepare", wraps=dataset.dataset.prepare) as prepare:
        dataset.prepare()

    prepare.assert_called_once_with()
    assert len(dataset) == 1


def test_numpy_fsl_text_dataset_rejects_source_mixture_child(tmp_path: Path):
    token_path = tmp_path / "tokens.npy"
    _write_mmap(token_path, [10, 11, 12, 13], np.uint16)
    child = NumpyFSLDatasetMixture(
        token_path,
        path_offset_index={(str(token_path), 0): 4},
        seed=1,
        sequence_length=4,
        pad_token_id=0,
        eos_token_id=1,
        vocab_size=32_000,
    )

    with pytest.raises(TypeError, match="complete semantic fingerprint"):
        NumpyFSLTextDataset(child)

    config = NumpyFSLTextDatasetConfig(dataset=Mock(source_mixture_config=object()))
    with pytest.raises(ValueError, match="does not support source_mixture_config"):
        config.build()


def test_numpy_fsl_text_dataset_rejects_document_lengths(tmp_path: Path):
    token_path = tmp_path / "tokens.npy"
    _write_mmap(token_path, [10, 11, 12, 13], np.uint16)
    child = NumpyFSLDataset(
        token_path,
        sequence_length=4,
        pad_token_id=0,
        eos_token_id=1,
        vocab_size=32_000,
        generate_doc_lengths=True,
    )

    with pytest.raises(ValueError, match="generate doc_lens"):
        NumpyFSLTextDataset(child)

    config = NumpyFSLTextDatasetConfig(
        dataset=NumpyFSLDatasetConfig(
            paths=[str(token_path)],
            sequence_length=4,
            tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=1, pad_token_id=0),
            generate_doc_lengths=True,
        )
    )
    with pytest.raises(ValueError, match="generate_doc_lengths=True"):
        config.build()
