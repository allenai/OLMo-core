"""What ``.edullm/train_on_corpus.py`` refuses, and what it lets through.

The entry point is not importable as a package -- it sits in ``.edullm/`` because that is what
the platform's image build copies and runs -- so it is loaded by path here. The alternative
was to leave the file untested, and the things it checks are precisely the ones that produce a
working run on wrong data rather than an error.

``edullm_data`` is not installed in this repository's CI, only in the built image. That is why
the module imports the reader inside ``resolve_corpus`` and why everything tested below is
reachable without it.
"""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pytest


def _load():
    path = Path(__file__).parent.parent.parent / ".edullm" / "train_on_corpus.py"
    spec = importlib.util.spec_from_file_location("edullm_train_on_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


entry = _load()


@dataclass
class FakeManifest:
    """The shape ``edullm_data.read.dataset_paths`` returns, with the fields that matter.

    Defaults describe a healthy corpus -- headerless, little-endian, uint32 -- so each test
    below changes exactly the one thing it is about.
    """

    paths: List[str] = field(default_factory=lambda: ["s3://edullm-data/x/v1/tokens/a.u32le.bin"])
    dtype: Optional[str] = "uint32"
    byte_order: Optional[str] = "little"
    header_bytes: int = 0
    rows: Optional[int] = 1000


def resolve(manifest: FakeManifest, tokenizer: str = "tokenizer/dolma2-bpe"):
    return entry.corpus_from_manifest(
        manifest, dataset_id="pretrain/regmix-10b", version="v1", tokenizer_id=tokenizer
    )


def test_a_healthy_corpus_keeps_the_width_the_manifest_declared():
    # The whole reason this file exists. OLMo-core's own fallback would look at dolma2's
    # 100,278-token vocab, conclude uint16 fits it, and read a uint32 corpus two bytes at a
    # time -- producing in-range ids, no error, and a loss curve that is merely bad.
    corpus = resolve(FakeManifest())
    assert str(corpus.dtype) == "uint32"
    assert corpus.tokenizer.vocab_size == 100278


def test_a_corpus_with_a_header_is_refused_rather_than_read_from_offset_zero():
    with pytest.raises(SystemExit, match="header"):
        resolve(FakeManifest(header_bytes=128))


def test_a_big_endian_corpus_is_refused_on_a_little_endian_host():
    other = "big" if sys.byteorder == "little" else "little"
    with pytest.raises(SystemExit, match="endian"):
        resolve(FakeManifest(byte_order=other))


def test_a_corpus_that_declares_no_width_is_refused_rather_than_guessed_at():
    with pytest.raises(SystemExit, match="no dtype"):
        resolve(FakeManifest(dtype=None))


def test_no_trainable_shards_is_an_error_and_not_an_empty_run():
    # A corpus whose only splits are held out resolves to nothing. Training on zero shards is
    # not a shorter run, it is a run whose loss means nothing.
    with pytest.raises(SystemExit, match="no trainable shards"):
        resolve(FakeManifest(paths=[]))


def test_an_unknown_tokenizer_names_the_ones_this_image_has():
    # Rather than defaulting. A default here trains on ids that mean something other than what
    # they meant when the corpus was tokenized, and nothing downstream can tell.
    with pytest.raises(SystemExit, match="dolma2-bpe"):
        resolve(FakeManifest(), tokenizer="tokenizer/bytes-utf8")


def test_the_whole_config_builds_from_a_corpus_without_touching_s3(monkeypatch):
    """The check that would otherwise cost an A10G to run.

    Every mistake in the config below -- a TrainerConfig field that was renamed, a callback
    argument that does not exist, a model factory that is gone -- raises here in a second.
    Without this, the first thing that discovers it is a twelve-hour submission that reached a
    GPU, pulled a three-gigabyte image, and died before the first step.
    """
    monkeypatch.setattr(
        entry,
        "resolve_corpus",
        lambda **kwargs: entry.corpus_from_manifest(
            FakeManifest(),
            dataset_id=kwargs["dataset_id"],
            version=kwargs["version"],
            tokenizer_id=kwargs["tokenizer_id"],
        ),
    )
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=s3://outputs/teams/platform/runs/a-run-id/checkpoints/",
            "--steps=25",
        ]
    )
    config = entry.build_config(opts, overrides)

    assert config.dataset.dtype == "uint32"
    assert config.dataset_id == "pretrain/regmix-10b"
    assert config.trainer.save_folder.endswith("/a-run-id/checkpoints/")
    # A retry must resume rather than overwrite what the first attempt left, which is the only
    # thing that makes a second attempt cheaper than a second run.
    assert config.trainer.save_overwrite is False
    # Pruning is off because the workload role has no delete permission. At OLMo-core's
    # default of three, the fourth save fails a run that is most of a day old.
    assert config.trainer.callbacks["checkpointer"].max_checkpoints is None
    # Serializing is what the config saver does beside the checkpoint; a config that cannot be
    # written is one whose record of what ran does not exist.
    assert config.as_config_dict()["dataset_version"] == "v1"


def test_an_override_on_the_command_line_reaches_the_config(monkeypatch):
    # The escape hatch researchers actually use: everything after the flags is merged into the
    # config, so a person can change the learning rate without a new entry point.
    monkeypatch.setattr(
        entry,
        "resolve_corpus",
        lambda **kwargs: entry.corpus_from_manifest(
            FakeManifest(),
            dataset_id=kwargs["dataset_id"],
            version=kwargs["version"],
            tokenizer_id=kwargs["tokenizer_id"],
        ),
    )
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=/tmp/x",
            "train_module.compile_model=false",
        ]
    )
    config = entry.build_config(opts, overrides)
    assert config.train_module.compile_model is False


def test_the_platform_variables_are_required_and_named_when_missing(monkeypatch, capsys):
    for name in (
        "EDULLM_DATASET_ID",
        "EDULLM_DATASET_VERSION",
        "EDULLM_DATASET_TOKENIZER",
        "EDULLM_CHECKPOINT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["train_on_corpus", "some-run"])

    with pytest.raises(SystemExit) as refusal:
        entry.main()

    # Naming all four at once rather than the first one missing: a person fixing a submission
    # should not have to discover them one failed run at a time.
    message = str(refusal.value)
    for name in ("EDULLM_DATASET_ID", "EDULLM_CHECKPOINT_DIR"):
        assert name in message
