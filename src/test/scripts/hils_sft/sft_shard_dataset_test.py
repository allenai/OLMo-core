"""Tests for the veomni SFT shard adapter (``memexpress/hils_sft/sft_shard_dataset.py``).

These assert the properties whose violation would still train, still converge, and still produce a
plausible eval number -- just for the wrong objective: loss on prompt tokens, loss on padding, or
labels that do not correspond to their inputs.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "scripts", "train", "memexpress", "hils_sft"),
)

from sft_shard_dataset import (  # noqa: E402
    IGNORE_INDEX,
    SFTShardDataset,
    mix_documents,
    split_documents,
)


def _pool(n_docs, doclen):
    return [
        (np.arange(doclen, dtype=np.uint32), np.ones(doclen, dtype=bool)) for _ in range(n_docs)
    ]

EOS, PAD = 100257, 100277

# (prompt ids, response ids) per document -- the mask is False on the prompt, True on the response
# and its terminating EOS, exactly as convert_unified_to_sft.py emits.
DOCS = [
    ([10, 11, 12], [20, 21]),
    ([13, 14], [22, 23, 24]),
    ([15], [25]),
]


def _write_shard(tmp_path, docs=DOCS, part=0):
    ids, mask = [], []
    for prompt, response in docs:
        ids.extend(prompt + response + [EOS])
        mask.extend([False] * len(prompt) + [True] * (len(response) + 1))
    np.save(tmp_path / f"token_ids_part_{part:06d}.npy", np.array(ids, dtype=np.uint32))
    np.save(tmp_path / f"labels_mask_{part:06d}.npy", np.array(mask, dtype=bool))
    return np.array(ids), np.array(mask)


def test_split_documents_splits_on_eos(tmp_path):
    ids, mask = _write_shard(tmp_path)
    docs = split_documents(ids, mask, EOS)
    assert len(docs) == len(DOCS)
    assert [len(d[0]) for d in docs] == [len(p) + len(r) + 1 for p, r in DOCS]
    for (doc_ids, _), (_, response) in zip(docs, DOCS):
        assert doc_ids[-1] == EOS
        assert doc_ids[-2] == response[-1]


def test_split_documents_drops_unterminated_tail(tmp_path):
    """A trailing document with no EOS is a truncated response -- a wrong target, so it is dropped."""
    ids = np.array([10, 20, EOS, 11, 21], dtype=np.uint32)
    mask = np.array([False, True, True, False, True], dtype=bool)
    assert len(split_documents(ids, mask, EOS)) == 1


def test_split_documents_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        split_documents(np.zeros(4, dtype=np.uint32), np.zeros(3, dtype=bool), EOS)


def test_missing_mask_twin_raises(tmp_path):
    np.save(tmp_path / "token_ids_part_000000.npy", np.array([10, 20, EOS], dtype=np.uint32))
    # Training on a token shard with no mask would put the PROMPT in the loss and never error.
    with pytest.raises(FileNotFoundError, match="labels_mask"):
        SFTShardDataset(str(tmp_path), 16, EOS, PAD)


def test_no_shards_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SFTShardDataset(str(tmp_path), 16, EOS, PAD)


@pytest.mark.parametrize("max_seq_len", [16, 32])
def test_labels_match_inputs_only_on_responses(tmp_path, max_seq_len):
    _write_shard(tmp_path)
    ds = SFTShardDataset(str(tmp_path), max_seq_len, EOS, PAD, seed=0)
    prompt_ids = {t for p, _ in DOCS for t in p}
    for i in range(len(ds)):
        ex = ds[i]
        assert len(ex["input_ids"]) == max_seq_len
        assert len(ex["labels"]) == max_seq_len
        trainable = ex["labels"] != IGNORE_INDEX
        # every trainable label equals its input id
        assert (ex["labels"][trainable] == ex["input_ids"][trainable]).all()
        # no prompt token is ever trainable
        assert not (set(ex["input_ids"][trainable].tolist()) & prompt_ids)
        # padding never contributes loss
        pad_positions = ex["input_ids"] == PAD
        assert (ex["labels"][pad_positions] == IGNORE_INDEX).all()


def test_padding_marked_in_attention_mask(tmp_path):
    _write_shard(tmp_path)
    ds = SFTShardDataset(str(tmp_path), 32, EOS, PAD, seed=0)
    ex = ds[0]
    content = int(ex["attention_mask"].sum())
    assert content == sum(len(p) + len(r) + 1 for p, r in DOCS)
    assert (ex["attention_mask"][content:] == 0).all()


def test_documents_longer_than_window_are_dropped_and_counted(tmp_path):
    _write_shard(tmp_path, docs=DOCS + [(list(range(50, 90)), [99])])
    ds = SFTShardDataset(str(tmp_path), 16, EOS, PAD, seed=0)
    stats = ds.stats()
    assert stats["documents_total"] == len(DOCS) + 1
    assert stats["documents_dropped_too_long"] == 1


def test_same_seed_gives_identical_batches(tmp_path):
    """The control that makes the two arms comparable: identical data, identical order."""
    _write_shard(tmp_path)
    a = SFTShardDataset(str(tmp_path), 16, EOS, PAD, seed=34521)
    b = SFTShardDataset(str(tmp_path), 16, EOS, PAD, seed=34521)
    assert len(a) == len(b)
    for i in range(len(a)):
        assert (a[i]["input_ids"] == b[i]["input_ids"]).all()
        assert (a[i]["labels"] == b[i]["labels"]).all()


# --- weighted mixing -------------------------------------------------------------------------
# The mixture is defined by sampling weights. On a flat concatenated corpus those weights would
# silently do nothing -- the realized share would just be each task's raw token count -- so these
# assert the weights actually bind.

def test_mix_hits_target_token_shares():
    per_source = {"contra": _pool(50, 100), "nq": _pool(2000, 100), "oolong": _pool(400, 100)}
    weights = {"contra": 2.9, "nq": 1.0, "oolong": 1.3}
    _, report = mix_documents(per_source, weights, seed=34521)
    for name, r in report.items():
        assert abs(r["realized_share"] - r["target_share"]) < 0.02, (name, r)


def test_mix_respects_repetition_cap():
    """A token-poor but heavily-weighted source must not be repeated past the cap."""
    per_source = {"contra": _pool(10, 100), "nq": _pool(5000, 100)}
    _, report = mix_documents(per_source, {"contra": 2.9, "nq": 1.0}, seed=0, max_repetition_factor=8.0)
    assert all(r["repetition_factor"] <= 8.0 + 1e-6 for r in report.values())
    # the scarce source is what binds the budget, so it should sit AT the cap
    assert report["contra"]["repetition_factor"] == pytest.approx(8.0, rel=1e-3)


def test_mix_is_deterministic_across_arms():
    per_source = {"a": _pool(30, 50), "b": _pool(60, 50)}
    w = {"a": 1.0, "b": 2.0}
    d1, _ = mix_documents(per_source, w, seed=34521)
    d2, _ = mix_documents(per_source, w, seed=34521)
    assert len(d1) == len(d2)
    assert all((x[0] == y[0]).all() for x, y in zip(d1, d2))


def test_mix_rejects_source_weight_mismatch():
    with pytest.raises(ValueError, match="disagree"):
        mix_documents({"a": _pool(5, 10), "b": _pool(5, 10)}, {"a": 1.0}, seed=0)


def test_mix_rejects_empty_source():
    with pytest.raises(ValueError, match="no tokens"):
        mix_documents({"a": _pool(5, 10), "b": []}, {"a": 1.0, "b": 1.0}, seed=0)


def test_dataset_sources_without_weights_raises(tmp_path):
    d = tmp_path / "s1"
    d.mkdir()
    _write_shard(d)
    with pytest.raises(ValueError, match="weights"):
        SFTShardDataset(str(d), 16, EOS, PAD, sources={"s1": str(d)})


def test_dataset_multi_source_reports_mixture(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_shard(a)
    _write_shard(b, docs=DOCS * 3)
    ds = SFTShardDataset(
        str(a), 32, EOS, PAD, sources={"a": str(a), "b": str(b)}, weights={"a": 3.0, "b": 1.0}
    )
    assert set(ds.mix_report) == {"a", "b"}
    assert ds.mix_report["a"]["target_share"] == pytest.approx(0.75)
    assert abs(ds.mix_report["a"]["realized_share"] - 0.75) < 0.05


# --- materialization: the one artifact all three arms read ------------------------------------
# The arms span two trainers whose data stacks do not agree. "Same data" therefore has to mean one
# materialized pack, not one recipe run twice -- two mixers and two packers give different windows.

def test_materialize_round_trips_byte_identically(tmp_path):
    from sft_shard_dataset import materialize

    src, out = tmp_path / "src", tmp_path / "pack"
    src.mkdir()
    _write_shard(src, docs=DOCS * 20)
    built = SFTShardDataset(str(src), 16, EOS, PAD, seed=34521)
    manifest = materialize(built, str(out), shard_windows=3)

    assert manifest["windows"] == len(built)
    assert manifest["tokens"] == len(built) * 16

    reread = SFTShardDataset(str(out), 16, EOS, PAD, prepacked=True)
    assert len(reread) == len(built)
    for i in range(len(built)):
        assert (reread[i]["input_ids"] == built[i]["input_ids"]).all(), i
        assert (reread[i]["labels"] == built[i]["labels"]).all(), i


def test_materialized_shards_are_window_aligned(tmp_path):
    """olmo_core recovers these windows by fixed-length chunking, which requires exact alignment."""
    from sft_shard_dataset import materialize

    src, out = tmp_path / "src", tmp_path / "pack"
    src.mkdir()
    _write_shard(src, docs=DOCS * 20)
    built = SFTShardDataset(str(src), 16, EOS, PAD, seed=34521)
    materialize(built, str(out), shard_windows=3)

    import glob

    for p in glob.glob(str(out / "token_ids_part_*.npy")):
        assert len(np.load(p)) % 16 == 0, p
    # and chunking the concatenated stream at the window length reproduces the windows in order
    stream = np.concatenate([np.load(p) for p in sorted(glob.glob(str(out / "token_ids_part_*.npy")))])
    for i in range(len(built)):
        assert (stream[i * 16 : (i + 1) * 16] == built[i]["input_ids"].numpy()).all(), i


def test_prepacked_rejects_misaligned_window(tmp_path):
    """Reading a pack at the wrong sequence_length must fail loudly, not silently re-window it."""
    from sft_shard_dataset import materialize

    src, out = tmp_path / "src", tmp_path / "pack"
    src.mkdir()
    _write_shard(src, docs=DOCS * 20)
    materialize(SFTShardDataset(str(src), 16, EOS, PAD, seed=0), str(out), shard_windows=3)
    with pytest.raises(ValueError, match="multiple of max_seq_len"):
        SFTShardDataset(str(out), 15, EOS, PAD, prepacked=True)
