from types import SimpleNamespace

import numpy as np
import pytest
import torch

from olmo_core.data import TokenizerConfig
from olmo_core.data.multimodal.collator import MultimodalCollatorConfig
from olmo_core.data.multimodal.mixture_data_loader import MixtureDataLoaderConfig


def text_example(length=3):
    return {
        "input_ids": np.arange(length, dtype=np.int64),
        "labels": np.arange(1, length + 1, dtype=np.int64),
        "loss_masks": np.ones(length, dtype=np.float32),
        "position_ids": np.arange(length, dtype=np.int64),
        "token_type_ids": np.zeros(length, dtype=np.int64),
        "images": np.empty((0, 4, 3), dtype=np.float32),
        "pooled_patches_idx": np.empty((0, 2), dtype=np.int64),
        "subsegment_ids": np.ones(length, dtype=np.int64),
        "example_ids": np.zeros(length, dtype=np.int64),
        "pack_source_names": ["text"],
    }


def test_text_only_preserves_tokens_masks_and_packing_metadata():
    examples = [text_example(), text_example(5)]
    default = MultimodalCollatorConfig(pad_token_id=7, pad_sequence_length=8).build()(examples)
    config = MultimodalCollatorConfig(pad_token_id=7, pad_sequence_length=8, text_only=True)
    assert MultimodalCollatorConfig.from_dict(config.as_config_dict()) == config
    text = config.build()(examples)

    assert set(default) - set(text) == {"images", "pooled_patches_idx"}
    assert default["images"].shape == (2, 1, 4, 3)
    assert torch.all(default["pooled_patches_idx"] == -1)
    assert torch.all(text["image_crop_counts"] == 0)
    assert torch.all(text["pooled_token_counts"] == 0)
    for name, value in text.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, default[name], rtol=0, atol=0)
        else:
            assert value == default[name]


@pytest.mark.parametrize("field", ["images", "pooled_patches_idx"])
def test_text_only_rejects_visual_data(field):
    example = text_example()
    original = example[field]
    example[field] = np.zeros((1, *original.shape[1:]), dtype=original.dtype)
    collator = MultimodalCollatorConfig(pad_token_id=0, text_only=True).build()
    with pytest.raises(ValueError, match="text-only collator"):
        collator([text_example(), example])


def test_text_only_does_not_allocate_dummy_image_arrays(monkeypatch):
    example = text_example()
    original_zeros = np.zeros

    def zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 4:
            pytest.fail("Text-only collation must not allocate a dummy image batch")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", zeros)
    batch = MultimodalCollatorConfig(pad_token_id=0, text_only=True).build()([example])
    assert "images" not in batch


@pytest.mark.parametrize("text_only", [False, True])
def test_mixture_loader_passes_explicit_text_only_policy(tmp_path, text_only):
    dataset = SimpleNamespace(
        datasets=[[text_example()]],
        weights=[1.0],
        names=["text"],
        tokenizer=TokenizerConfig.dolma2(),
    )
    config = MixtureDataLoaderConfig(
        global_batch_size=8,
        sequence_length=8,
        work_dir=str(tmp_path),
        text_only=text_only,
    )
    assert MixtureDataLoaderConfig.from_dict(config.as_config_dict()) == config
    loader = config.build(dataset)
    assert loader.collator.text_only is text_only
    batch = loader.collator([text_example()])
    assert ("images" not in batch) is text_only
