"""
The training entry points' argument surface.

Only the pure half is testable here -- the recipe needs olmo-core and a GPU. But the argument
surface is where a launch goes wrong cheaply and silently: a mis-parsed mix weight, a budget that
was never set, a landmark sequence length that is not a multiple of the block size. Each of those
produces a job that starts, burns node hours and is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TRAIN = Path(__file__).parents[3] / "src" / "scripts" / "ctc" / "train"
sys.path.insert(0, str(TRAIN))

options = pytest.importorskip("options", reason="training scripts not importable")


def _opts(*extra, mode="sft"):
    argv = ["run-name", "--data", "/shards/a", "--base", "/base", "--max-steps", "100", *extra]
    args = options.build_parser("t", mode=mode).parse_args(argv)
    return options.options_from_args(args, mode=mode)


# ── the data mix ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spec,path,weight,name",
    [
        ("/s/contra", "/s/contra", 1.0, "contra"),
        ("/s/contra:2", "/s/contra", 2.0, "contra"),
        ("/s/contra:2:contradiction", "/s/contra", 2.0, "contradiction"),
        ("/s/contra/", "/s/contra/", 1.0, "contra"),
    ],
)
def test_data_spec_parsing(spec, path, weight, name):
    parsed = options.parse_data_spec(spec)
    assert (parsed.path, parsed.weight, parsed.name) == (path, weight, name)


@pytest.mark.parametrize("bad", ["/s/a:notanumber", "/s/a:0", "/s/a:-1"])
def test_a_bad_weight_is_rejected(bad):
    with pytest.raises(ValueError):
        options.parse_data_spec(bad)


def test_weights_are_ratios_not_probabilities():
    """'--data a:2 --data b:1' has to mean 2:1 whatever the numbers' scale."""
    opts = _opts("--data", "/shards/b:3")
    assert [round(r, 6) for _, r in opts.weights()] == [0.25, 0.75]

    doubled = _opts("--data", "/shards/b:3")
    doubled.data = [options.DataSpec(d.path, d.weight * 10, d.label) for d in doubled.data]
    assert [round(r, 6) for _, r in doubled.weights()] == [0.25, 0.75]


# ── guards ──────────────────────────────────────────────────────────────────────────────────────


def test_a_budget_must_be_given_exactly_once():
    """Neither means the run never stops; both means one is silently ignored."""
    with pytest.raises(ValueError, match="exactly one"):
        options.TrainOptions(run_name="r", data=[options.DataSpec("/a")])
    with pytest.raises(ValueError, match="exactly one"):
        options.TrainOptions(run_name="r", data=[options.DataSpec("/a")], max_steps=1, max_tokens=1)


def test_a_landmark_seq_len_must_fit_whole_blocks():
    with pytest.raises(ValueError, match="multiple of the landmark block size"):
        options.TrainOptions(
            run_name="r",
            data=[options.DataSpec("/a")],
            max_steps=1,
            arch="landmark",
            seq_len=41000,  # 41000/64 = 640.625
            mem_freq=63,
        )
    options.TrainOptions(
        run_name="r",
        data=[options.DataSpec("/a")],
        max_steps=1,
        arch="landmark",
        seq_len=40960,
        mem_freq=63,
    )


def test_an_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="--arch must be one of"):
        options.TrainOptions(run_name="r", data=[options.DataSpec("/a")], max_steps=1, arch="nope")


def test_omitting_the_base_requires_saying_so():
    """
    Training from random init when you meant to fine-tune produces a run that looks healthy and
    means nothing, so it cannot be the accidental default.
    """
    args = options.build_parser("t", mode="sft").parse_args(
        ["r", "--data", "/a", "--max-steps", "1"]
    )
    with pytest.raises(SystemExit, match="from-scratch"):
        options.options_from_args(args, mode="sft")

    args.from_scratch = True
    assert options.options_from_args(args, mode="sft").base is None


def test_the_missing_base_message_mentions_the_marker_repair():
    """A fresh Qwen3 base has bit-identical marker embeddings; training doc-chunked from one
    flatlines and reads as a modelling result."""
    args = options.build_parser("t", mode="sft").parse_args(
        ["r", "--data", "/a", "--max-steps", "1"]
    )
    with pytest.raises(SystemExit, match="marker embeddings"):
        options.options_from_args(args, mode="sft")


# ── defaults that differ by mode ────────────────────────────────────────────────────────────────


def test_sft_and_cpt_differ_only_in_their_defaults():
    assert options.build_parser("t", mode="sft").get_default("lr") == 1e-5
    assert options.build_parser("t", mode="cpt").get_default("lr") == 1e-4

    sft_flags = {a.dest for a in options.build_parser("t", mode="sft")._actions}
    cpt_flags = {a.dest for a in options.build_parser("t", mode="cpt")._actions}
    assert sft_flags == cpt_flags, "the two entry points must expose the same options"


def test_the_fingerprint_callback_is_on_by_default():
    assert _opts().fingerprint is True
    assert _opts("--no-fingerprint").fingerprint is False


def test_batch_size_follows_the_world_size():
    opts = _opts("--nodes", "4")
    assert opts.world_size == 32
    assert opts.global_batch_tokens == 32 * opts.seq_len


def test_a_cluster_makes_it_a_beaker_run():
    assert _opts().is_local is True
    assert _opts("--cluster", "ai2/jupiter-cirrascale-2").is_local is False


def test_describe_names_everything_a_launch_should_be_auditable_by():
    text = options.describe(_opts("--data", "/shards/b:3", "--arch", "chunked"))
    for expected in ("chunked", "/base", "run-name", "100 steps", "fingerprint on"):
        assert expected in text
