"""
The train/eval format guard.

Each test below stands for a bug that reached a results table. They are written as "this specific
historical mistake now raises", because a guard that is merely plausible is a guard nobody trusts
enough to leave enabled.
"""

from __future__ import annotations

import json

import pytest

from ctc.format.fingerprint import (
    FINGERPRINT_FILENAME,
    FormatFingerprint,
    FormatMismatchError,
    check_or_explain_missing,
    hash_prompt,
)


def make(**overrides) -> FormatFingerprint:
    base = dict(
        task="contradiction",
        prompt_hash=hash_prompt("INSTRUCTION", "Claim {id}: {text}"),
        serializer="contradiction",
        item_separator="\n\n",
        gold_index_base=1,
        chunk_layout="wrap_documents",
        doc_id_range=(1, 697),
        marker_token_ids=(151648, 151649),
        tokenizer="Qwen3.5-4B",
    )
    base.update(overrides)
    return FormatFingerprint(**base)


# ── the historical failures ─────────────────────────────────────────────────────────────────────

def test_identical_formats_are_compatible():
    assert make().compare(make()) == []
    make().require_compatible_with(make())


def test_doc_id_digit_range_beyond_training_is_caught():
    """Train saw ids up to 697, eval shows up to 1423. This read as long-context collapse."""
    trained = make(doc_id_range=(1, 697))
    evaluating = make(doc_id_range=(1, 1423))
    with pytest.raises(FormatMismatchError) as e:
        evaluating.require_compatible_with(trained)
    assert [m.field for m in e.value.mismatches] == ["doc_id_range"]


def test_doc_id_range_inside_training_is_fine():
    """Containment, not equality -- a shorter eval ladder is legitimate."""
    make(doc_id_range=(1, 100)).require_compatible_with(make(doc_id_range=(1, 697)))


def test_gold_index_base_flip_is_caught():
    """contradiction is 1-based; outlier/rerank/nq are 0-based. Off by one on every answer."""
    with pytest.raises(FormatMismatchError):
        make(gold_index_base=0).require_compatible_with(make(gold_index_base=1))


def test_chunk_layout_drift_is_caught():
    with pytest.raises(FormatMismatchError):
        make(chunk_layout="none").require_compatible_with(make(chunk_layout="wrap_documents"))


def test_marker_token_ids_drift_is_caught():
    """Shards built against one marker id set, checkpoint repaired against another."""
    with pytest.raises(FormatMismatchError):
        make(marker_token_ids=(1, 2)).require_compatible_with(make(marker_token_ids=(3, 4)))


def test_item_separator_drift_is_caught():
    """A single newline puts items in the wrong chunk under the chunked/landmark masks."""
    with pytest.raises(FormatMismatchError):
        make(item_separator="\n").require_compatible_with(make(item_separator="\n\n"))


def test_an_edited_instruction_is_caught():
    """Editing an instruction string makes the checkpoint's format obsolete."""
    with pytest.raises(FormatMismatchError):
        make(prompt_hash=hash_prompt("INSTRUCTION but reworded")).require_compatible_with(make())


# ── the error is actionable ─────────────────────────────────────────────────────────────────────

def test_error_names_every_bad_field_not_just_the_first():
    """A bare hash would say 'something differs'; bisecting that is the cost this avoids."""
    evaluating = make(gold_index_base=0, chunk_layout="none", item_separator="\n")
    with pytest.raises(FormatMismatchError) as e:
        evaluating.require_compatible_with(make())
    assert {m.field for m in e.value.mismatches} == {
        "gold_index_base", "chunk_layout", "item_separator"
    }
    text = str(e.value)
    for f in ("gold_index_base", "chunk_layout", "item_separator"):
        assert f in text


def test_notes_are_recorded_but_never_compared():
    a = make(notes={"built_by": "alice", "date": "2026-01-01"})
    b = make(notes={"built_by": "bob"})
    assert a.compare(b) == []


# ── hashing ─────────────────────────────────────────────────────────────────────────────────────

def test_hash_is_order_sensitive():
    assert hash_prompt("a", "b") != hash_prompt("b", "a")


def test_hash_distinguishes_absent_from_empty():
    """'no template' and 'empty template' are different formats."""
    assert hash_prompt("a", None) != hash_prompt("a", "")


def test_hash_is_stable_across_processes():
    """sha256, not builtin hash() -- PYTHONHASHSEED would otherwise make every run mismatch."""
    assert hash_prompt("INSTRUCTION") == hash_prompt("INSTRUCTION")


# ── validation ──────────────────────────────────────────────────────────────────────────────────

def test_bad_gold_index_base_is_rejected_at_construction():
    with pytest.raises(ValueError, match="gold_index_base"):
        make(gold_index_base=2)


def test_inverted_doc_id_range_is_rejected():
    with pytest.raises(ValueError, match="inverted"):
        make(doc_id_range=(50, 10))


def test_schema_version_mismatch_refuses_to_compare():
    """Treating an absent field as 'matches' is how a guard stops guarding."""
    with pytest.raises(ValueError, match="schema"):
        make().compare(make(schema_version=99))


# ── i/o ─────────────────────────────────────────────────────────────────────────────────────────

def test_round_trip(tmp_path):
    fp = make()
    fp.write(tmp_path)
    assert FormatFingerprint.read(tmp_path) == fp


def test_tuples_survive_json(tmp_path):
    """JSON has no tuples; reading back a list would fail the containment comparison."""
    make().write(tmp_path)
    got = FormatFingerprint.read(tmp_path)
    assert isinstance(got.doc_id_range, tuple)
    assert isinstance(got.marker_token_ids, tuple)


def test_read_returns_none_for_a_directory_without_one(tmp_path):
    assert FormatFingerprint.read(tmp_path) is None


def _write_raw(tmp_path, *records):
    (tmp_path / FINGERPRINT_FILENAME).write_text(
        json.dumps({"schema_version": 1, "formats": list(records)})
    )


def test_unknown_field_is_rejected_rather_than_dropped(tmp_path):
    """An old reader must not validate a newer record it cannot fully understand."""
    _write_raw(tmp_path, {**make().to_dict(), "attention_variant": "landmark"})
    with pytest.raises(ValueError, match="unknown field"):
        FormatFingerprint.read(tmp_path)


def test_truncated_record_raises_rather_than_matching(tmp_path):
    _write_raw(tmp_path, {"task": "contradiction"})
    with pytest.raises(TypeError):
        FormatFingerprint.read(tmp_path)


def test_a_bare_record_is_not_mistaken_for_a_set(tmp_path):
    """Accepting one would silently lose every other task in a mix's record."""
    (tmp_path / FINGERPRINT_FILENAME).write_text(json.dumps(make().to_dict()))
    with pytest.raises(ValueError, match="not a fingerprint set"):
        FormatFingerprint.read(tmp_path)


# ── the enforcement helper ──────────────────────────────────────────────────────────────────────

def test_strict_mode_refuses_an_unfingerprinted_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="UNVERIFIED"):
        check_or_explain_missing(make(), tmp_path, strict=True)


def test_non_strict_returns_a_warning_to_record(tmp_path):
    """Old checkpoints have no fingerprint; grading them is allowed but must be disclosed."""
    warning = check_or_explain_missing(make(), tmp_path, strict=False)
    assert warning is not None and "UNVERIFIED" in warning


def test_helper_raises_on_a_real_mismatch(tmp_path):
    make(doc_id_range=(1, 697)).write(tmp_path)
    with pytest.raises(FormatMismatchError):
        check_or_explain_missing(make(doc_id_range=(1, 1423)), tmp_path)


def test_helper_passes_a_matching_checkpoint(tmp_path):
    make().write(tmp_path)
    assert check_or_explain_missing(make(), tmp_path) is None
