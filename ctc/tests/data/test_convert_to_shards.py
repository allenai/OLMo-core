"""
The JSONL -> shards converter, for the parts that need neither olmo-core nor a GPU.

Two things are worth testing here without a tokenizer: the startup guards, each of which stands in
for a defect that required a full data rebuild, and the format fingerprint the converter writes --
which is the link that lets the eval-side guard verify anything at all about a trained checkpoint.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

from ctc.format import registry
from ctc.format.fingerprint import chunk_layout_for
from ctc.tasks import load_all

SCRIPTS = Path(__file__).parents[3] / "src" / "scripts" / "ctc"
sys.path.insert(0, str(SCRIPTS))

convert = pytest.importorskip("convert_to_shards", reason="converter module not importable")


# ── chunk_layout vocabulary ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "emit,chunk_by,markers,expected",
    [
        ("dense", "document", True, "wrap_documents"),
        ("dense", "line", True, "wrap_lines"),
        ("landmark", "document", True, "landmark_documents"),
        ("landmark", "line", True, "landmark_lines"),
        ("dense", "document", False, "none"),
        ("landmark", "line", False, "none"),
    ],
)
def test_chunk_layout_names(emit, chunk_by, markers, expected):
    assert chunk_layout_for(emit, chunk_by, markers) == expected


def test_a_marker_free_build_records_one_layout_regardless_of_chunk_by():
    """
    Without boundary tokens the chunking is not observable in the stream, so recording it would
    make two byte-identical shards compare as different formats.
    """
    assert chunk_layout_for("dense", "document", False) == chunk_layout_for(
        "landmark", "line", False
    )


@pytest.mark.parametrize("bad", [("nope", "document"), ("dense", "nope")])
def test_an_unknown_layout_option_is_rejected(bad):
    with pytest.raises(ValueError):
        chunk_layout_for(bad[0], bad[1], True)


# ── the item-regex guard ────────────────────────────────────────────────────────────────────────


def test_the_bare_pipe_regex_is_rejected():
    """
    '||' is an alternation of two empty branches, so it matches every line: the instruction and
    header lines each become their own chunk and the blanks between them stay FREE, bridging chunks
    and mismatching the eval layout. This silently produced the oolong chunk leak.
    """
    with pytest.raises(SystemExit, match="EMPTY STRING"):
        convert.check_item_regex("||")


def test_the_escaped_regex_is_accepted():
    convert.check_item_regex(r"\|\|")


def test_an_invalid_regex_is_rejected_with_its_own_message():
    with pytest.raises(SystemExit, match="not a valid regex"):
        convert.check_item_regex("[unclosed")


@pytest.mark.parametrize("pattern", ["", "a*", "(?:)", "x?"])
def test_every_empty_matching_pattern_is_rejected(pattern):
    """The guard is on 'matches the empty string', not on the literal string '||'."""
    with pytest.raises(SystemExit, match="EMPTY STRING"):
        convert.check_item_regex(pattern)


# ── the fingerprint written beside the shards ───────────────────────────────────────────────────


def _args(**overrides):
    base = dict(
        task="contradiction",
        emit="dense",
        chunk_by="document",
        query_position="both",
        tokenizer="Qwen/Qwen3.5-4B-Base",
        cot_mode="none",
        use_titles=False,
        no_doc_markers=False,
        input_jsonl=["/data/ctc/v3/contradiction/train.jsonl"],
    )
    base.update(overrides)
    return Namespace(**base)


class Ids:
    doc_start, doc_end, eos, landmark, pad = 248049, 248050, 248044, 248200, 248203


@pytest.fixture(scope="module")
def examples():
    return [
        {
            "documents": [{"text": f"Claim {i}."} for i in range(n)],
            "queries": [""],
            "answers": [],
            "gold_doc_indices": [[1, 2]],
        }
        for n in (20, 40)
    ]


def test_the_fingerprint_records_the_options_actually_used(examples):
    load_all()
    fp = convert.build_fingerprint(_args(), Ids, examples)

    assert fp.task == "contradiction"
    assert fp.query_position == "both"
    assert fp.chunk_layout == "wrap_documents"
    assert fp.marker_token_ids == (248049, 248050)
    assert fp.tokenizer == "Qwen/Qwen3.5-4B-Base"


def test_the_landmark_build_records_the_landmark_and_pad_ids(examples):
    """A landmark shard inserts two more reserved ids than a dense one, so a checkpoint trained on
    one must not be graded against the other's token stream."""
    fp = convert.build_fingerprint(_args(emit="landmark"), Ids, examples)
    assert fp.marker_token_ids == (248049, 248050, 248200, 248203)
    assert fp.chunk_layout == "landmark_documents"


def test_a_marker_free_build_records_no_marker_ids(examples):
    fp = convert.build_fingerprint(_args(no_doc_markers=True), Ids, examples)
    assert fp.marker_token_ids is None
    assert fp.chunk_layout == "none"


def test_doc_id_range_is_measured_from_the_training_data(examples):
    """
    The only place this can honestly be measured. Taken from an eval file it would contain itself
    by construction -- a check that passes because it was derived from the thing being checked.
    """
    fp = convert.build_fingerprint(_args(), Ids, examples)
    assert fp.doc_id_range == (1, 40)
    assert fp.notes["provenance"] == "measured"
    assert fp.notes["measured_over"] == 2


def test_the_input_files_are_recorded_as_data_paths(examples):
    fp = convert.build_fingerprint(_args(), Ids, examples)
    assert fp.data_paths == (str(Path("/data/ctc/v3/contradiction/train.jsonl").resolve()),)


def test_the_written_fingerprint_round_trips_and_guards(tmp_path, examples):
    """
    End to end for the link this converter exists to create: write beside the shards, read back,
    and refuse an eval whose format differs. This is the whole chain minus the trainer.
    """
    from ctc.format.fingerprint import FingerprintSet, FormatMismatchError

    trained = convert.build_fingerprint(_args(), Ids, examples)
    FingerprintSet([trained]).write(tmp_path)

    recovered = FingerprintSet.read(tmp_path)
    assert recovered is not None and recovered.tasks == ["contradiction"]

    spec = registry.get("contradiction")
    recovered.require_compatible(
        spec.fingerprint(
            query_position="both",
            chunk_layout="wrap_documents",
            marker_token_ids=(248049, 248050),
            tokenizer="Qwen/Qwen3.5-4B-Base",
            doc_id_range=(1, 20),
        )
    )
    with pytest.raises(FormatMismatchError, match="query_position"):
        recovered.require_compatible(
            spec.fingerprint(
                query_position="after",
                chunk_layout="wrap_documents",
                marker_token_ids=(248049, 248050),
                tokenizer="Qwen/Qwen3.5-4B-Base",
            )
        )


def test_the_attention_mode_is_not_part_of_the_format(tmp_path, examples):
    """
    The full-vs-chunked comparison runs over identically tokenized shards -- "full" is a mask, not a
    token layout, and the box markers are present in both arms. So a wrap_documents checkpoint
    graded with --attn full must PASS the guard; only a marker-free prompt is a real mismatch.
    """
    from ctc.format.fingerprint import FingerprintSet, FormatMismatchError

    trained = convert.build_fingerprint(_args(), Ids, examples)
    recovered = FingerprintSet([trained])
    spec = registry.get("contradiction")

    common = dict(
        query_position="both",
        marker_token_ids=(248049, 248050),
        tokenizer="Qwen/Qwen3.5-4B-Base",
    )
    recovered.require_compatible(spec.fingerprint(chunk_layout="wrap_documents", **common))
    with pytest.raises(FormatMismatchError, match="chunk_layout"):
        recovered.require_compatible(spec.fingerprint(chunk_layout="none", **common))
