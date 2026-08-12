"""
The ``--base`` preflight in ``src/scripts/ctc/train/run.py``.

``load_strategy=always`` already guarantees a missing base fails rather than silently training from
random init -- but it fails from inside ``trainer.fit()``, after both ranks have built the model and
FSDP has wrapped it. On a real multi-rank run that is a couple of minutes of a GPU allocation spent
learning that a path was missing its ``model_and_optim`` component, which is exactly the mistake the
marker-repair script's output layout invites: it writes ``<out>/model_and_optim/``, and ``<out>``
itself is not something olmo-core recognises as a checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("olmo_core", reason="needs olmo-core")

TRAIN_SCRIPTS = Path(__file__).parents[3] / "src" / "scripts" / "ctc" / "train"
if str(TRAIN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRAIN_SCRIPTS))

run_mod = pytest.importorskip("run", reason="train/run.py not importable")


def make_checkpoint(root: Path) -> Path:
    """The minimal thing ``Checkpointer.dir_is_checkpoint`` accepts: a bare ``.metadata``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".metadata").write_bytes(b"")
    return root


def test_a_real_checkpoint_dir_passes(tmp_path):
    run_mod._check_base_is_a_checkpoint(str(make_checkpoint(tmp_path / "ckpt")))


def test_the_parent_of_model_and_optim_is_refused_and_the_message_names_the_fix(tmp_path):
    base = tmp_path / "base_fixmark"
    make_checkpoint(base / "model_and_optim")

    with pytest.raises(SystemExit) as e:
        run_mod._check_base_is_a_checkpoint(str(base))

    msg = str(e.value)
    assert "model_and_optim" in msg
    assert str(base / "model_and_optim") in msg, "the error must print the path that would work"


def test_a_directory_that_is_no_checkpoint_at_all_is_refused_without_a_bogus_hint(tmp_path):
    base = tmp_path / "junk"
    base.mkdir()
    (base / "notes.txt").write_text("hi")

    with pytest.raises(SystemExit) as e:
        run_mod._check_base_is_a_checkpoint(str(base))
    assert "Pass:" not in str(e.value), "no hint should be offered when there is nothing to point at"


def test_a_nonlocal_path_is_left_to_olmo_core(tmp_path):
    # s3://, gs://, or a weka path not mounted on the launch host: not our business to validate.
    run_mod._check_base_is_a_checkpoint("s3://bucket/some/checkpoint")
    run_mod._check_base_is_a_checkpoint(str(tmp_path / "does-not-exist"))
