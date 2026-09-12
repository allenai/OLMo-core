import json
from dataclasses import dataclass, field

import numpy as np
import pytest

from olmo_core.config import Config
from olmo_core.data.multimodal.alignment import (
    MultimodalMixtureConfig,
    MultimodalSourceConfig,
)
from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds, prepare_molmo2_tokenizer


@dataclass
class ExampleDatasetConfig(Config):
    size: int = 10
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)

    def build(self, tokenizer):
        return ExampleDataset(self)


class ExampleDataset:
    def __init__(self, config):
        self.config = config
        self.content_fingerprint = "example-dataset"

    def __len__(self):
        return self.config.size

    def __getitem__(self, index):
        return self.get(index, 0)

    def get(self, index, epoch=0):
        return {"row": index, "epoch": epoch}

    def raw_image_references(self, index):
        return (f"image-{index}",)


def test_config_round_trip_and_nested_overrides():
    config = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig.dolma2(),
        sources={
            "new_captions": MultimodalSourceConfig(
                dataset=PixMoCapDatasetConfig(dataset_path="synthetic", mode="caption")
            )
        },
        target_loss_mass={"new_captions": 1.0},
        mean_loss_weight={"new_captions": 12.0},
    )
    restored = MultimodalMixtureConfig.from_dict(config.as_config_dict())
    assert restored == config
    updated = restored.merge(["sources.new_captions.dataset.max_crops=2"])
    assert isinstance(updated.sources["new_captions"], MultimodalSourceConfig)
    assert isinstance(updated.sources["new_captions"].dataset, PixMoCapDatasetConfig)
    assert updated.sources["new_captions"].dataset.max_crops == 2
    assert config.sources["new_captions"].dataset.max_crops == 8


def test_build_retains_order_and_calibrates_loss_mass():
    config = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig.dolma2(),
        sources={"z_caption": ExampleDatasetConfig(), "a_count": ExampleDatasetConfig()},
        target_loss_mass={"a_count": 0.25, "z_caption": 0.75},
        mean_loss_weight={"z_caption": 3.0, "a_count": 1.0},
    )
    token_ids = Molmo2TokenIds(im_start_id=7)
    mixture = config.build(tokenizer=object(), token_ids=token_ids)
    assert mixture.names == ["z_caption", "a_count"]
    assert mixture.weights == pytest.approx([0.5, 0.5])
    for dataset in mixture.datasets:
        assert dataset.config.token_ids == token_ids
    assert config.sources["z_caption"].token_ids != token_ids


def test_build_sources_allows_preparing_uncalibrated_mixture():
    config = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig.dolma2(),
        sources={"new_source": ExampleDatasetConfig()},
        target_loss_mass={"new_source": 1.0},
    )
    assert len(config.build_sources(object(), Molmo2TokenIds())["new_source"]) == 10
    with pytest.raises(ValueError, match="must not be empty"):
        config.build(tokenizer=object(), token_ids=Molmo2TokenIds())
    config.mean_loss_weight = {"retired_source": 2.0}
    with pytest.raises(ValueError, match="source mismatch"):
        config.build(tokenizer=object(), token_ids=Molmo2TokenIds())


@pytest.mark.parametrize("suffix", [".txt", ".npy"])
def test_selection_preserves_order_epoch_and_image_references(tmp_path, suffix):
    selection = tmp_path / f"indices{suffix}"
    if suffix == ".npy":
        np.save(selection, np.asarray([7, 2, 4], dtype=np.int64))
    else:
        selection.write_text("7\n2\n4\n")
    config = MultimodalSourceConfig(dataset=ExampleDatasetConfig(), selection_path=str(selection))
    dataset = config.build(object())
    assert len(dataset) == 3
    assert dataset[0] == {"row": 7, "epoch": 0}
    assert dataset.get(1, 5) == {"row": 2, "epoch": 5}
    assert dataset.raw_image_references(2) == ("image-4",)
    with pytest.raises(IndexError):
        dataset[-1]
    with pytest.raises(IndexError):
        dataset[3]


def test_disjoint_selection_check(tmp_path):
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "validation.txt"
    train_path.write_text("1\n2\n3\n")
    val_path.write_text("4\n5\n")
    config = MultimodalSourceConfig(
        dataset=ExampleDatasetConfig(),
        selection_path=str(train_path),
        excluded_selection_paths=[str(val_path)],
    )
    assert len(config.build(object())) == 3
    val_path.write_text("3\n4\n")
    with pytest.raises(ValueError, match="overlaps excluded selection"):
        config.build(object())


@pytest.mark.parametrize(
    "contents,error",
    [("1\n1\n", "duplicate"), ("-1\n", "invalid"), ("10\n", "outside"), ("", "nonempty")],
)
def test_invalid_selection_is_rejected(tmp_path, contents, error):
    selection = tmp_path / "rows.txt"
    selection.write_text(contents)
    with pytest.raises(ValueError, match=error):
        MultimodalSourceConfig(dataset=ExampleDatasetConfig(), selection_path=str(selection)).build(
            object()
        )


def test_selected_identity_depends_on_rows_and_config_not_selection_filename(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("1\n3\n")
    second.write_text("1\n3\n")
    config = MultimodalSourceConfig(dataset=ExampleDatasetConfig(), selection_path=str(first))
    original = config.build(object()).content_fingerprint
    assert (
        config.replace(selection_path=str(second)).build(object()).content_fingerprint == original
    )
    second.write_text("3\n1\n")
    assert (
        config.replace(selection_path=str(second)).build(object()).content_fingerprint != original
    )
    config.dataset.size = 11
    assert config.build(object()).content_fingerprint != original


def test_build_tokenizer_uses_configured_identity(monkeypatch):
    from transformers import AutoTokenizer
    from transformers.models.auto import tokenization_auto

    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 3

        def __init__(self):
            self.vocab = {"<|im_end|>": 4}

        def add_tokens(self, tokens, special_tokens=False):
            for token in tokens:
                self.vocab.setdefault(token, len(self.vocab) + 4)

        def get_vocab(self):
            return self.vocab

        def convert_tokens_to_ids(self, token):
            return self.vocab[token]

    calls = []

    def load(identifier, **kwargs):
        calls.append((identifier, kwargs))
        return Tokenizer()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    monkeypatch.setattr(tokenization_auto, "get_tokenizer_config", lambda *args, **kwargs: {})
    config = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig(
            vocab_size=5, eos_token_id=2, pad_token_id=3, identifier="parent"
        ),
        model_vocab_size=16,
        tokenizer_revision="pinned-revision",
    )
    tokenizer, ids = config.build_tokenizer()
    assert tokenizer.eos_token_id == 2
    assert ids.im_start_id == 5
    assert calls[0] == (
        "parent",
        {"config": None, "revision": "pinned-revision", "cache_dir": None, "use_fast": False},
    )
    config.tokenizer.pad_token_id = 8
    with pytest.raises(ValueError, match="pad_token_id"):
        config.build_tokenizer()


@pytest.mark.parametrize("self_describing", [True, False])
def test_build_tokenizer_from_offline_cache(tmp_path, monkeypatch, self_describing):
    from huggingface_hub import constants
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import GPT2Config, GPT2Tokenizer
    from transformers.utils import hub

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(hub, "is_offline_mode", lambda: True)
    revision = "a" * 40
    snapshot = tmp_path / "models--test--tokenizer" / "snapshots" / revision
    vocab = {token: index for index, token in enumerate(sorted(ByteLevel.alphabet()))}
    expected = GPT2Tokenizer(vocab=vocab, merges=[], pad_token="<pad>")
    expected.add_tokens(["<|im_end|>"], special_tokens=True)
    expected.save_pretrained(snapshot)
    assert not (snapshot / "config.json").exists()
    if not self_describing:
        metadata_path = snapshot / "tokenizer_config.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.pop("tokenizer_class")
        metadata_path.write_text(json.dumps(metadata))
        GPT2Config().save_pretrained(snapshot)

    config = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig(
            identifier="test/tokenizer",
            vocab_size=len(expected),
            eos_token_id=expected.eos_token_id,
            pad_token_id=expected.pad_token_id,
        ),
        tokenizer_revision=revision,
        tokenizer_cache_dir=str(tmp_path),
        model_vocab_size=len(expected) + 6,
    )
    actual, token_ids = config.build_tokenizer()
    expected_ids = prepare_molmo2_tokenizer(expected, model_vocab_size=len(expected) + 6)
    assert type(actual) is type(expected)
    assert actual.get_vocab() == expected.get_vocab()
    assert actual.special_tokens_map == expected.special_tokens_map
    assert token_ids == expected_ids
    texts = ["Hello world", " café 中文 🦉", "def f(x):\n    return x + 1", "<|im_end|><im_patch>"]
    assert actual(texts)["input_ids"] == expected(texts)["input_ids"]


def test_calibration_is_bounded_reproducible_and_does_not_mutate(monkeypatch):
    class CalibrationDataset:
        def __init__(self, size):
            self.size = size
            self.calls = []

        def __len__(self):
            return self.size

        def get(self, index, epoch):
            self.calls.append((index, epoch))
            return {"loss_masks": [0.0, float(index + 1)]}

    datasets = {"large": CalibrationDataset(100000000), "small": CalibrationDataset(3)}
    config = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig.dolma2(), mean_loss_weight={"old_source": 2.0}
    )
    monkeypatch.setattr(config, "build_tokenizer", lambda: (object(), Molmo2TokenIds()))
    monkeypatch.setattr(config, "build_sources", lambda *args: datasets)
    first = config.estimate_mean_loss_weights(8, seed=41)
    assert first["small"] == 2.0
    assert len(datasets["large"].calls) == len(set(datasets["large"].calls)) == 8
    assert len(datasets["small"].calls) == 3
    assert all(epoch == 0 for _, epoch in datasets["large"].calls)
    first_indices = datasets["large"].calls[:]
    datasets = {name: datasets[name] for name in reversed(datasets)}
    second = config.estimate_mean_loss_weights(8, seed=41)
    assert first == second
    assert datasets["large"].calls[8:] == first_indices
    assert config.mean_loss_weight == {"old_source": 2.0}


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_calibration_rejects_invalid_loss_weight(monkeypatch, weight):
    config = MultimodalMixtureConfig(tokenizer=TokenizerConfig.dolma2())
    monkeypatch.setattr(config, "build_tokenizer", lambda: (object(), Molmo2TokenIds()))
    monkeypatch.setattr(
        config, "build_sources", lambda *args: {"invalid_source": [{"loss_masks": [weight]}]}
    )
    with pytest.raises(ValueError, match="invalid_source.*loss weight"):
        config.estimate_mean_loss_weights()
