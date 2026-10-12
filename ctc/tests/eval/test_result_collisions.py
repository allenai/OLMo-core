"""
A result file is never silently replaced by a different measurement.

Result names are ``<task>_<rung>_<attn>[_<tag>].json``. They carry neither the checkpoint, nor the
bundle, nor the query position -- so a second pass over the same checkpoint with another bundle, or
the same bundle on another checkpoint sharing an output directory, lands on the first pass's
filenames and the survivor is indistinguishable from the loser. Nothing about the file afterwards
says which pass wrote it.

The name itself cannot be widened: the eval skills, the launch ledger and the results-hub ingest all
read these names by hand. So the collision is caught instead, up front and by name, and a rerun of
the *same* pass still overwrites -- which is a thing people legitimately do all day.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctc.eval import cli

BUNDLE = "/bundles/v2_clean"
CKPT = "/ckpt/run-a/step1100"


def _todo(task="contradiction", rung="2k"):
    return [{"task": task, "spec": task, "rung": rung, "path": Path(f"/data/{task}_{rung}.jsonl")}]


def _write_result(path: Path, *, bundle=BUNDLE, query_position="both", ckpt=CKPT) -> Path:
    """Write a result file shaped like a real one, for the fields identity is read from."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ladder": "contradiction",
                "bundle_root": bundle,
                "provenance": {"ckpt": ckpt, "query_position": query_position},
                "_meta": {"eval_bundle": bundle, "query_position": query_position},
            }
        )
    )
    return path


def _collisions(out_dir, *, bundle=BUNDLE, query_position="both", ckpt=CKPT, tag=""):
    return cli.collisions(
        _todo(), Path(out_dir), "full", tag, cli._identity(bundle, query_position, ckpt)
    )


# ── the filename pattern is a contract with other tooling ───────────────────────────────────────


@pytest.mark.parametrize(
    "task,rung,attn,tag,expected",
    [
        ("contradiction", "2k", "full", "", "contradiction_2k_full.json"),
        ("contradiction", "8k", "chunked", "", "contradiction_8k_chunked.json"),
        ("nq", "4k", "landmark", "fast", "nq_4k_landmark_fast.json"),
    ],
)
def test_the_result_filename_pattern_is_unchanged(task, rung, attn, tag, expected):
    """Documented as ``<task>_<rung>_<attn>[_<tag>].json`` in both eval skills. Readers glob it."""
    item = {"task": task, "rung": rung}
    assert cli._result_path(Path("/out"), item, attn, tag) == Path("/out") / expected


# ── what counts as a collision ──────────────────────────────────────────────────────────────────


def test_nothing_on_disk_is_not_a_collision(tmp_path):
    assert _collisions(tmp_path) == []


def test_rerunning_the_same_pass_still_overwrites(tmp_path):
    """Same checkpoint, same bundle, same layout: a rerun, not a second measurement."""
    _write_result(tmp_path / "contradiction_2k_full.json")
    assert _collisions(tmp_path) == []


@pytest.mark.parametrize(
    "field,changed",
    [
        ("bundle_root", {"bundle": "/bundles/v2"}),
        ("query_position", {"query_position": "after"}),
        ("ckpt", {"ckpt": "/ckpt/run-b/step900"}),
    ],
)
def test_a_different_measurement_is_a_collision(tmp_path, field, changed):
    _write_result(tmp_path / "contradiction_2k_full.json")
    report = _collisions(tmp_path, **changed)
    assert len(report) == 1
    assert field in report[0]


def test_the_report_names_both_values(tmp_path):
    """'Impossible to miss' means the message says what is there and what would replace it."""
    _write_result(tmp_path / "contradiction_2k_full.json", bundle="/bundles/fast")
    (block,) = _collisions(tmp_path)
    assert str(tmp_path / "contradiction_2k_full.json") in block
    assert "/bundles/fast" in block
    assert BUNDLE in block


def test_a_distinguishing_tag_avoids_the_collision_entirely(tmp_path):
    """The documented remedy has to actually work: a tagged pass writes a different file."""
    _write_result(tmp_path / "contradiction_2k_full.json", bundle="/bundles/fast")
    assert _collisions(tmp_path, tag="fast") == []


def test_a_trailing_slash_is_not_a_difference(tmp_path):
    _write_result(tmp_path / "contradiction_2k_full.json")
    assert _collisions(tmp_path, bundle=BUNDLE + "/", ckpt=CKPT + "/") == []


# ── files that cannot answer the question ───────────────────────────────────────────────────────


def test_a_result_predating_the_identity_fields_does_not_block(tmp_path):
    """A file that records nothing about its bundle is not evidence of a collision either way."""
    path = tmp_path / "contradiction_2k_full.json"
    path.write_text(json.dumps({"ladder": "contradiction", "metrics": {"f1": 0.5}}))
    assert _collisions(tmp_path) == []


def test_a_partial_result_is_compared_on_what_it_does_record(tmp_path):
    """Absent fields are skipped; the ones present are still checked."""
    path = tmp_path / "contradiction_2k_full.json"
    path.write_text(json.dumps({"provenance": {"ckpt": "/ckpt/run-b/step900"}}))
    (block,) = _collisions(tmp_path)
    assert "ckpt" in block and "bundle_root" not in block


@pytest.mark.parametrize("junk", ["not json at all", "[]", '"a string"'])
def test_an_unreadable_file_is_not_treated_as_a_collision(tmp_path, junk):
    """Refusing to run because the directory holds something else would be its own failure mode."""
    (tmp_path / "contradiction_2k_full.json").write_text(junk)
    assert _collisions(tmp_path) == []


def test_query_position_is_read_from_provenance_when_meta_is_absent(tmp_path):
    path = tmp_path / "contradiction_2k_full.json"
    path.write_text(json.dumps({"provenance": {"ckpt": CKPT, "query_position": "before"}}))
    (block,) = _collisions(tmp_path)
    assert "query_position" in block


# ── end to end through the CLI ──────────────────────────────────────────────────────────────────


def _argv(tmp_path, bundle, *extra):
    return [
        "--ckpt",
        CKPT,
        "--tasks",
        "contradiction",
        "--rungs",
        "2k",
        "--bundle",
        bundle,
        "--out",
        str(tmp_path),
        *extra,
    ]


def test_the_cli_refuses_before_the_checkpoint_loads(tmp_path, monkeypatch):
    """The check is worth nothing after a rung that took hours, so it runs before the model does."""
    _write_result(tmp_path / "contradiction_2k_full.json", bundle="/bundles/v2")
    monkeypatch.setattr(
        cli.backends, "load", lambda *a, **k: pytest.fail("the backend must not be loaded")
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main(_argv(tmp_path, BUNDLE))
    message = str(excinfo.value)
    assert "would be overwritten" in message
    assert "/bundles/v2" in message and BUNDLE in message
    # Every documented way out is named, because the right one depends on what the caller meant.
    assert "--tag" in message and "--out" in message and "--overwrite" in message


class _BackendLoaded(Exception):
    """Raised by the stub backend loader: reaching it means the guard let the run through."""


def test_overwrite_is_the_way_past_it(tmp_path, monkeypatch):
    _write_result(tmp_path / "contradiction_2k_full.json", bundle="/bundles/v2")

    def _stub(*args, **kwargs):
        raise _BackendLoaded

    monkeypatch.setattr(cli.backends, "load", _stub)
    monkeypatch.setattr(cli.backends, "available", lambda: ["native"])
    # The run gets as far as loading the checkpoint, which is exactly as far as this test can go.
    with pytest.raises(_BackendLoaded):
        cli.main(_argv(tmp_path, BUNDLE, "--overwrite"))


def test_overwrite_is_off_by_default():
    assert cli.build_parser().parse_args(["--ckpt", "x"]).overwrite is False
