"""
The training recipe assembles, for every architecture and both modes.

Needs olmo-core but not a GPU: building configs is CPU work, and it is where the cheap mistakes
live. The first run of this caught a real one -- ``ConcatAndChunkInstanceSourceConfig`` takes no
``tokenizer`` argument, so every CPT launch would have died at config time, which no amount of
reading the SFT path would have revealed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("olmo_core.data.composable", reason="needs olmo-core")

TRAIN = Path(__file__).parents[3] / "src" / "scripts" / "ctc" / "train"
sys.path.insert(0, str(TRAIN))

options = pytest.importorskip("options")
recipe = pytest.importorskip("recipe")

ARCHES = options.ARCHITECTURES


def _opts(arch, mode, **extra):
    budget = {"max_steps": 10} if mode == "sft" else {"max_tokens": 1000}
    return options.TrainOptions(
        run_name="t",
        data=[options.DataSpec("/shards/a", 2), options.DataSpec("/shards/b", 1)],
        base="/base",
        arch=arch,
        seq_len=8192,
        mode=mode,
        **budget,
        **extra,
    )


@pytest.mark.parametrize("mode", ["sft", "cpt"])
@pytest.mark.parametrize("arch", ARCHES)
def test_every_architecture_and_mode_assembles(tmp_path, arch, mode):
    model, train_module, dataset, loader, trainer = recipe.build_experiment(
        _opts(arch, mode), save_folder=str(tmp_path / "run"), work_dir=str(tmp_path / "work")
    )
    assert model is not None and train_module is not None and trainer is not None
    expected = "PadToLength" if mode == "sft" else "ConcatAndChunk"
    assert type(dataset).__name__.startswith(expected)


@pytest.mark.parametrize("arch", [a for a in ARCHES if a != "full"])
def test_the_document_masks_get_their_marker_ids(arch):
    """Without these the mask cannot derive chunk roles and silently degrades to plain causal."""
    model = recipe.build_model_config(_opts(arch, "sft"), vocab_size=152064)
    ids = model.document_chunk_attention
    assert ids["doc_start_id"] and ids["doc_end_id"] and ids["eos_id"]
    assert ("pad_id" in ids) == (arch == "landmark"), "only the landmark layout fills windows"


def test_plain_causal_gets_no_chunk_config():
    model = recipe.build_model_config(_opts("full", "sft"), vocab_size=152064)
    assert not getattr(model, "document_chunk_attention", None)


def test_the_fingerprint_callback_is_wired_from_the_shard_dirs(tmp_path):
    """
    Collected from the data, not declared in the launcher: a launcher's declaration is a claim
    about the data, and when they disagree the launcher is the one that is wrong.
    """
    trainer = recipe.build_trainer_config(
        _opts("chunked", "sft"), save_folder=str(tmp_path), work_dir=str(tmp_path)
    )
    callback = trainer.callbacks["format_fingerprint"]
    assert callback.collect_from == ["/shards/a", "/shards/b"]

    off = recipe.build_trainer_config(
        _opts("chunked", "sft", fingerprint=False),
        save_folder=str(tmp_path),
        work_dir=str(tmp_path),
    )
    assert "format_fingerprint" not in off.callbacks


def test_the_base_checkpoint_is_loaded_weights_only(tmp_path):
    """Inheriting the base's optimizer state or step count would make the schedule and every
    logged step number meaningless for a new run over new data."""
    trainer = recipe.build_trainer_config(
        _opts("full", "sft"), save_folder=str(tmp_path), work_dir=str(tmp_path)
    )
    assert trainer.load_path == "/base"
    assert trainer.load_optim_state is False
    assert trainer.load_trainer_state is False
