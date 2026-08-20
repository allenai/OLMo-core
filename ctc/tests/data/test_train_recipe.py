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


def _tiny_shards(directory: Path, *, mode: str, instances: int = 4, length: int = 64) -> str:
    """Write the smallest shard pair the loader will accept.

    :param directory: Where to write them.
    :param mode: ``"sft"`` writes a label mask; ``"cpt"`` does not read one.
    :param instances: How many EOS-terminated instances.
    :param length: Tokens per instance.

    :returns: The directory, as a string.
    """
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    ids = np.arange(instances * length, dtype=np.uint32) % 1000
    ids[length - 1 :: length] = 151643  # EOS terminates every instance
    ids.tofile(directory / "token_ids_part_000000.npy")
    if mode == "sft":
        mask = np.zeros(instances * length, dtype=np.bool_)
        mask[length // 2 :: length] = True  # some answer tokens, or there is no loss
        mask.tofile(directory / "labels_mask_part_000000.npy")
    return str(directory)


@pytest.mark.parametrize("mode", ["sft", "cpt"])
def test_the_instance_source_actually_builds_from_shards(tmp_path, mode):
    """`build_experiment` returns a source CONFIG, and the loader's `build` takes a built
    InstanceSource. Asserting only on the config's type -- as the test above does -- let a wiring
    bug through: `run.py` handed the config straight to the loader and died on a GPU with
    `TypeError: object of type 'PadToLengthInstanceSourceConfig' has no len()`. Build it here, on
    CPU, where the mistake is free to find."""
    shards = _tiny_shards(tmp_path / "shards", mode=mode)
    opts = options.TrainOptions(
        run_name="t",
        data=[options.DataSpec(shards)],
        base="/base",
        arch="full",
        seq_len=64,
        mode=mode,
        **({"max_steps": 2} if mode == "sft" else {"max_tokens": 256}),
    )
    work = tmp_path / "work"
    _, _, dataset, loader, _ = recipe.build_experiment(
        opts, save_folder=str(tmp_path / "run"), work_dir=str(work)
    )
    source = dataset.build(str(work))
    assert len(source) > 0, "a built source must report its instance count"
    assert source.sequence_length == 64


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


def test_chunked_mix_installs_an_annealed_curriculum():
    """
    The reference "chunked" numbers were trained with this curriculum (p 0.80 -> 0.0), so getting
    it silently wrong reproduces a different experiment. The anneal length must equal the
    per-rank forward count -- this recipe runs one padded instance per rank per step, so that is
    the step budget; dividing by world size a second time is the historical bug where p stalled
    at ``mix_start_p * (1 - 1/world_size)``.
    """
    def opts(**budget):
        return options.TrainOptions(
            run_name="t",
            data=[options.DataSpec("/shards/a", 1)],
            base="/base",
            arch="chunked-mix",
            seq_len=8192,
            nodes=4,
            mode="sft",
            **budget,
        )

    model = recipe.build_model_config(opts(max_steps=1100), vocab_size=152064)
    mix = model.document_chunk_attention
    assert mix["mix_start_p"] == 0.80
    assert mix["mix_end_p"] == 0.0
    assert mix["mix_total_forwards"] == 1100
    # A token budget anneals over the implied step count instead: 100 steps of 32 instances of
    # 8192 tokens (4 nodes x 8 GPUs, one instance per rank per step).
    tokens = recipe.build_model_config(opts(max_tokens=8192 * 32 * 100), vocab_size=152064)
    assert tokens.document_chunk_attention["mix_total_forwards"] == 100


def test_plain_chunked_gets_no_mix_keys():
    """`chunked` must stay bit-identical to the pure mask: p == 0 is a different arm, not a
    default of the same one."""
    model = recipe.build_model_config(_opts("chunked", "sft"), vocab_size=152064)
    assert "mix_start_p" not in model.document_chunk_attention
