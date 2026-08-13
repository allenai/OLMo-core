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
    split_documents,
)

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
