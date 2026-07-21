"""
Tests for the landmark gate-analysis hook (:mod:`olmo_core.nn.attention.landmark_gate_analysis`),
focused on the gate-*score* capture: each kept gate is logged with its raw landmark logit, the two
parallel ``blocks`` / ``scores`` lists stay descending-by-score and one-per-block, and the optional
``OLMO_GATE_LOG_ALL`` mode additionally emits every candidate block's score.
"""

import json

import torch

from olmo_core.nn.attention import landmark_gate_analysis as gate_log


def _reset() -> None:
    """Clear the module's cached global state so each test re-reads the environment from scratch."""
    gate_log.close()
    gate_log._initialized = False
    gate_log._enabled = False
    gate_log._log_all = False
    gate_log._file = None
    gate_log._path = None
    gate_log._current_layers = {}
    gate_log._doc_counter = -1


def _read_record(monkeypatch, tmp_path, *, log_all: bool, keep, block_ids, scores):
    """Drive one example/one decode step through the hook and return the single JSONL record."""
    monkeypatch.setenv("OLMO_LANDMARK_GATE_LOG", str(tmp_path / "gate.jsonl"))
    if log_all:
        monkeypatch.setenv("OLMO_GATE_LOG_ALL", "1")
    else:
        monkeypatch.delenv("OLMO_GATE_LOG_ALL", raising=False)
    # No RANK under plain pytest, so the recorder suffixes the path with the pid; keep it unset.
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    _reset()
    try:
        assert gate_log.is_enabled()
        gate_log.start_example(content_prompt_len=100)
        gate_log.record_layer(0, keep, block_ids, scores)
        gate_log.finalize_token()
        path = gate_log._path
        assert path is not None
        gate_log.close()
        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
    finally:
        _reset()
    assert len(lines) == 1
    return lines[0]


def test_records_kept_blocks_with_scores(monkeypatch, tmp_path):
    # B=1, H=2, M=4 gate slots, one landmark block per slot.
    block_ids = torch.tensor([0, 1, 2, 3])
    keep = torch.tensor(
        [[[[True, False, True, True]], [[True, True, False, False]]]], dtype=torch.bool
    )  # (1, 2, 1, 4)
    scores = torch.tensor([[[[1.0, 5.0, 3.0, 2.0]], [[4.0, 7.0, 9.0, 8.0]]]])  # (1, 2, 1, 4)

    rec = _read_record(
        monkeypatch, tmp_path, log_all=False, keep=keep, block_ids=block_ids, scores=scores
    )

    assert rec["decoded_token_num"] == 1
    head0 = rec["layers"]["layer0"]["head0"]
    # Kept slots {0, 2, 3} ordered by descending score: block2 (3.0), block3 (2.0), block0 (1.0).
    assert head0["blocks"] == [2, 3, 0]
    assert head0["scores"] == [3.0, 2.0, 1.0]

    head1 = rec["layers"]["layer0"]["head1"]
    # Kept slots {0, 1} ordered by score: block1 (7.0), block0 (4.0). Slots 2/3 are not kept, so
    # their (higher) scores must NOT leak into the kept lists.
    assert head1["blocks"] == [1, 0]
    assert head1["scores"] == [7.0, 4.0]
    # Default mode does not emit the full distribution.
    assert "all_blocks" not in head0 and "all_scores" not in head0


def test_log_all_records_every_candidate_block(monkeypatch, tmp_path):
    block_ids = torch.tensor([0, 1, 2, 3])
    keep = torch.tensor([[[[True, False, True, True]]]], dtype=torch.bool)  # (1, 1, 1, 4)
    scores = torch.tensor([[[[1.0, 5.0, 3.0, 2.0]]]])

    rec = _read_record(
        monkeypatch, tmp_path, log_all=True, keep=keep, block_ids=block_ids, scores=scores
    )
    head0 = rec["layers"]["layer0"]["head0"]
    # Kept set is unchanged.
    assert head0["blocks"] == [2, 3, 0]
    assert head0["scores"] == [3.0, 2.0, 1.0]
    # all_* covers every slot (incl. the not-kept block1, the highest scorer), descending by score.
    assert head0["all_blocks"] == [1, 2, 3, 0]
    assert head0["all_scores"] == [5.0, 3.0, 2.0, 1.0]


def test_dedupes_shared_block_ordinal_keeping_best_slot(monkeypatch, tmp_path):
    # Two slots map to the same block (e.g. a multi-landmark chunk); the block appears once, scored
    # by its highest slot.
    block_ids = torch.tensor([0, 0, 1])
    keep = torch.tensor([[[[True, True, True]]]], dtype=torch.bool)
    scores = torch.tensor([[[[1.0, 9.0, 2.0]]]])

    rec = _read_record(
        monkeypatch, tmp_path, log_all=False, keep=keep, block_ids=block_ids, scores=scores
    )
    head0 = rec["layers"]["layer0"]["head0"]
    assert head0["blocks"] == [0, 1]
    assert head0["scores"] == [9.0, 2.0]


def test_non_finite_scores_are_dropped(monkeypatch, tmp_path):
    # Sparse-landmark absent chunks arrive as -inf; they must not appear in the output.
    block_ids = torch.tensor([0, 1])
    keep = torch.tensor([[[[True, True]]]], dtype=torch.bool)
    scores = torch.tensor([[[[float("-inf"), 3.0]]]])

    rec = _read_record(
        monkeypatch, tmp_path, log_all=True, keep=keep, block_ids=block_ids, scores=scores
    )
    head0 = rec["layers"]["layer0"]["head0"]
    assert head0["blocks"] == [1] and head0["scores"] == [3.0]
    assert head0["all_blocks"] == [1] and head0["all_scores"] == [3.0]
