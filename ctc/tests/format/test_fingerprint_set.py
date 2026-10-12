"""
Fingerprint sets: a checkpoint records every format it was trained under, not one.

The canonical SFT mix trains five tasks at once, so eval's question is narrower than "does this
checkpoint match" -- it is "was *this task* trained, and in the layout I am about to use". The
distinction matters most in its failure mode: a task that was never trained is not a mismatch, it
is an absence, and reporting it as a mismatch would send someone looking for drift that is not
there.
"""

from __future__ import annotations

import pytest

from ctc.format.fingerprint import (
    FINGERPRINT_FILENAME,
    FingerprintSet,
    FormatFingerprint,
    FormatMismatchError,
    TaskNotTrainedError,
    check_or_explain_missing,
    hash_prompt,
)


def make(task="contradiction", **overrides) -> FormatFingerprint:
    base = dict(
        task=task,
        prompt_hash=hash_prompt(f"{task.upper()} INSTRUCTION"),
        serializer=task,
        item_separator="\n\n",
        gold_index_base=1,
        chunk_layout="wrap_documents",
        tokenizer="Qwen3.5-4B",
    )
    base.update(overrides)
    return FormatFingerprint(**base)


MIX = ["contradiction", "outlier", "oolong", "retrieval", "nq"]


def a_mix() -> FingerprintSet:
    return FingerprintSet([make(t, gold_index_base=1 if t == "contradiction" else 0) for t in MIX])


# ── the mix ─────────────────────────────────────────────────────────────────────────────────────


def test_a_mix_records_every_task():
    assert a_mix().tasks == MIX


def test_a_trained_task_in_a_mix_is_checked_against_its_own_entry():
    a_mix().require_compatible(make("outlier", gold_index_base=0))


def test_a_task_in_the_mix_still_fails_on_a_real_mismatch():
    """The other four entries must not dilute the one that matters."""
    with pytest.raises(FormatMismatchError) as e:
        a_mix().require_compatible(make("outlier", gold_index_base=1))
    assert [m.field for m in e.value.mismatches] == ["gold_index_base"]


def test_an_untrained_task_is_an_absence_not_a_mismatch():
    with pytest.raises(TaskNotTrainedError) as e:
        a_mix().require_compatible(make("mathmatch", gold_index_base=1))
    assert e.value.task == "mathmatch"
    assert e.value.trained == MIX
    assert "out-of-distribution" in str(e.value)


# ── a task trained under two layouts ────────────────────────────────────────────────────────────


def test_either_of_two_recorded_layouts_is_accepted():
    """A curriculum can vary the layout on purpose; such a checkpoint is bound to both."""
    both = FingerprintSet([make(query_position="before"), make(query_position="after")])
    both.require_compatible(make(query_position="before"))
    both.require_compatible(make(query_position="after"))


def test_a_third_layout_is_still_refused():
    both = FingerprintSet([make(query_position="before"), make(query_position="after")])
    with pytest.raises(FormatMismatchError):
        both.require_compatible(make(query_position="both"))


def test_the_error_reports_the_closest_candidate_only():
    """Several unrelated mismatch lists would obscure which one nearly matched."""
    candidates = FingerprintSet(
        [
            make(query_position="before"),
            make(query_position="after", gold_index_base=0, item_separator="\n"),
        ]
    )
    with pytest.raises(FormatMismatchError) as e:
        candidates.require_compatible(make(query_position="both"))
    assert [m.field for m in e.value.mismatches] == ["query_position"]


# ── set algebra ─────────────────────────────────────────────────────────────────────────────────


def test_merge_is_a_union_that_drops_exact_duplicates():
    merged = FingerprintSet([make("a"), make("b")]).merge(FingerprintSet([make("b"), make("c")]))
    assert merged.tasks == ["a", "b", "c"]
    assert len(merged.formats) == 3


def test_merge_keeps_two_genuinely_different_formats_for_one_task():
    merged = FingerprintSet([make(query_position="before")]).merge(
        FingerprintSet([make(query_position="after")])
    )
    assert len(merged.for_task("contradiction")) == 2


def test_an_empty_set_is_refused():
    """Writing one would make an unfingerprinted checkpoint look fingerprinted."""
    with pytest.raises(ValueError, match="at least one"):
        FingerprintSet([])


def test_for_task_returns_nothing_for_an_absent_task():
    assert a_mix().for_task("mathmatch") == []


# ── i/o ─────────────────────────────────────────────────────────────────────────────────────────


def test_round_trip(tmp_path):
    a_mix().write(tmp_path)
    assert FingerprintSet.read(tmp_path) == a_mix()


def test_a_single_fingerprint_writes_a_set_of_one(tmp_path):
    """One on-disk shape, so a shard dir and a checkpoint dir are read the same way."""
    make().write(tmp_path)
    assert FingerprintSet.read(tmp_path).tasks == ["contradiction"]


def test_reading_a_multi_task_record_as_a_single_one_refuses(tmp_path):
    a_mix().write(tmp_path)
    with pytest.raises(ValueError, match="records 5 formats"):
        FormatFingerprint.read(tmp_path)


def test_write_replaces_rather_than_appending(tmp_path):
    a_mix().write(tmp_path)
    FingerprintSet([make("solo")]).write(tmp_path)
    assert FingerprintSet.read(tmp_path).tasks == ["solo"]


def test_read_returns_none_for_a_directory_without_one(tmp_path):
    assert FingerprintSet.read(tmp_path) is None


def test_the_filename_is_the_one_eval_looks_for(tmp_path):
    assert a_mix().write(tmp_path).name == FINGERPRINT_FILENAME


# ── the enforcement helper, through a set ───────────────────────────────────────────────────────


def test_helper_passes_a_task_in_the_mix(tmp_path):
    a_mix().write(tmp_path)
    assert check_or_explain_missing(make("outlier", gold_index_base=0), tmp_path) is None


def test_helper_raises_on_mismatch_even_when_not_strict(tmp_path):
    """Absence is a warning; an actual mismatch never is."""
    a_mix().write(tmp_path)
    with pytest.raises(FormatMismatchError):
        check_or_explain_missing(make("outlier", gold_index_base=1), tmp_path, strict=False)


def test_helper_downgrades_an_untrained_task_to_a_warning_when_not_strict(tmp_path):
    a_mix().write(tmp_path)
    warning = check_or_explain_missing(make("mathmatch"), tmp_path, strict=False)
    assert warning is not None and "UNVERIFIED" in warning


def test_helper_raises_on_an_untrained_task_when_strict(tmp_path):
    a_mix().write(tmp_path)
    with pytest.raises(TaskNotTrainedError):
        check_or_explain_missing(make("mathmatch"), tmp_path, strict=True)
