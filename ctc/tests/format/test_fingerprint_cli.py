"""
``ctc-fingerprint``.

The command exists mostly for checkpoints that predate the guard, which makes its riskiest
behaviour *writing*: a backfilled fingerprint is an assertion about how something was trained, and
a wrong assertion converts the guard from something that catches a mismatch into something that
certifies one. Several tests below are about making that asymmetry visible rather than convenient.
"""

from __future__ import annotations

import json

import pytest

from ctc.format.cli import main
from ctc.format.fingerprint import FINGERPRINT_FILENAME, FingerprintSet


def rung(tmp_path, name="rung_2048.jsonl", n_docs=(20, 40)):
    examples = [
        {
            "documents": [{"text": f"Claim {i}."} for i in range(n)],
            "queries": ["Find contradicting claims."],
            "answers": [""],
            "gold_doc_indices": [[1, 2]],
        }
        for n in n_docs
    ]
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(e) for e in examples) + "\n")
    return str(p)


def written(directory):
    return FingerprintSet.read(directory)


# ── write ───────────────────────────────────────────────────────────────────────────────────────


def test_write_records_the_task(tmp_path):
    assert main(["write", "--dir", str(tmp_path), "--task", "contradiction"]) == 0
    assert written(tmp_path).tasks == ["contradiction"]


def test_query_position_lands_in_the_record(tmp_path):
    """The dimension a reproduction run got wrong precisely because nothing recorded it."""
    main(["write", "--dir", str(tmp_path), "--task", "contradiction", "--query-position", "both"])
    assert written(tmp_path).formats[0].query_position == "both"


def test_doc_id_range_is_measured_from_the_data(tmp_path):
    """Measured, not claimed -- the digit-range bug was a wrong claim about the corpus."""
    main(["write", "--dir", str(tmp_path), "--task", "retrieval", "--data", rung(tmp_path)])
    fp = written(tmp_path).formats[0]
    assert fp.doc_id_range == (1, 40)
    assert fp.notes["provenance"] == "measured"


def test_an_asserted_range_is_labelled_as_asserted(tmp_path, capsys):
    main(["write", "--dir", str(tmp_path), "--task", "retrieval", "--doc-id-range", "1:697"])
    fp = written(tmp_path).formats[0]
    assert fp.doc_id_range == (1, 697)
    assert fp.notes["provenance"] == "asserted"
    assert "ASSERTED" in capsys.readouterr().out


def test_an_unnumbered_task_records_no_range(tmp_path):
    """oolong renders items verbatim; a range there would invent a constraint."""
    main(["write", "--dir", str(tmp_path), "--task", "oolong", "--data", rung(tmp_path)])
    assert written(tmp_path).formats[0].doc_id_range is None


def test_write_replaces_by_default(tmp_path):
    main(["write", "--dir", str(tmp_path), "--task", "contradiction"])
    main(["write", "--dir", str(tmp_path), "--task", "outlier"])
    assert written(tmp_path).tasks == ["outlier"]


def test_merge_accumulates_a_mix(tmp_path):
    main(["write", "--dir", str(tmp_path), "--task", "contradiction"])
    main(["write", "--dir", str(tmp_path), "--task", "outlier", "--merge"])
    assert written(tmp_path).tasks == ["contradiction", "outlier"]


def test_marker_ids_are_parsed(tmp_path):
    main(
        [
            "write",
            "--dir",
            str(tmp_path),
            "--task",
            "contradiction",
            "--marker-token-ids",
            "151648,151649",
        ]
    )
    assert written(tmp_path).formats[0].marker_token_ids == (151648, 151649)


# ── collect ─────────────────────────────────────────────────────────────────────────────────────


def test_collect_stamps_a_checkpoint_from_its_shard_dirs(tmp_path):
    a, b, ckpt = tmp_path / "a", tmp_path / "b", tmp_path / "ckpt"
    main(["write", "--dir", str(a), "--task", "contradiction"])
    main(["write", "--dir", str(b), "--task", "outlier"])
    assert main(["collect", "--ckpt", str(ckpt), "--from", str(a), str(b)]) == 0
    assert written(ckpt).tasks == ["contradiction", "outlier"]


def test_collect_refuses_an_unfingerprinted_source(tmp_path, capsys):
    a = tmp_path / "a"
    main(["write", "--dir", str(a), "--task", "contradiction"])
    rc = main(["collect", "--ckpt", str(tmp_path / "ckpt"), "--from", str(a), str(tmp_path / "x")])
    assert rc == 1
    assert not (tmp_path / "ckpt" / FINGERPRINT_FILENAME).exists()
    assert "out-of-distribution" in capsys.readouterr().err


def test_collect_can_be_forced_but_says_the_record_is_incomplete(tmp_path, capsys):
    a = tmp_path / "a"
    main(["write", "--dir", str(a), "--task", "contradiction"])
    rc = main(
        [
            "collect",
            "--ckpt",
            str(tmp_path / "ckpt"),
            "--from",
            str(a),
            str(tmp_path / "x"),
            "--allow-missing",
        ]
    )
    assert rc == 0
    assert "INCOMPLETE" in capsys.readouterr().out


# ── check ───────────────────────────────────────────────────────────────────────────────────────


def test_check_passes_a_matching_format(tmp_path):
    ckpt = tmp_path / "ckpt"
    main(["write", "--dir", str(ckpt), "--task", "contradiction", "--query-position", "both"])
    assert (
        main(["check", "--ckpt", str(ckpt), "--task", "contradiction", "--query-position", "both"])
        == 0
    )


def test_check_catches_the_query_position_mismatch_without_a_gpu(tmp_path, capsys):
    """The point of the subcommand: find this in a second rather than in a rung of decoding."""
    ckpt = tmp_path / "ckpt"
    main(["write", "--dir", str(ckpt), "--task", "contradiction", "--query-position", "both"])
    rc = main(
        ["check", "--ckpt", str(ckpt), "--task", "contradiction", "--query-position", "after"]
    )
    assert rc == 1
    assert "query_position" in capsys.readouterr().err


def test_check_reports_an_untrained_task_as_such(tmp_path, capsys):
    ckpt = tmp_path / "ckpt"
    main(["write", "--dir", str(ckpt), "--task", "contradiction"])
    assert main(["check", "--ckpt", str(ckpt), "--task", "outlier"]) == 1
    assert "out-of-distribution" in capsys.readouterr().err


def test_check_reports_an_unfingerprinted_checkpoint(tmp_path, capsys):
    assert main(["check", "--ckpt", str(tmp_path), "--task", "contradiction"]) == 1
    assert "UNVERIFIED" in capsys.readouterr().out


# ── show ────────────────────────────────────────────────────────────────────────────────────────


def test_show_surfaces_that_a_record_was_asserted(tmp_path, capsys):
    main(["write", "--dir", str(tmp_path), "--task", "retrieval", "--doc-id-range", "1:697"])
    capsys.readouterr()
    assert main(["show", str(tmp_path)]) == 0
    assert "asserted" in capsys.readouterr().out


def test_show_exits_nonzero_for_an_unfingerprinted_directory(tmp_path, capsys):
    assert main(["show", str(tmp_path)]) == 1
    assert "UNVERIFIED" in capsys.readouterr().out


def test_show_json_round_trips(tmp_path, capsys):
    main(["write", "--dir", str(tmp_path), "--task", "contradiction"])
    capsys.readouterr()
    main(["show", str(tmp_path), "--json"])
    assert json.loads(capsys.readouterr().out) == written(tmp_path).to_dict()


# ── argument handling ───────────────────────────────────────────────────────────────────────────


def test_an_unknown_task_is_rejected(tmp_path):
    with pytest.raises(KeyError):
        main(["write", "--dir", str(tmp_path), "--task", "not_a_task"])
