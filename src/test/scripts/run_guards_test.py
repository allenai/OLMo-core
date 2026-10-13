"""
Tests for the start-of-run guards in ``recipes/train/run_guards.py``.

Each guard stands in for a trap that has already produced a plausible-looking wrong result: a
``--max-tokens`` budget that silently trains one step, a relaunch that silently resumes another
run's checkpoint, and an unrepaired marker base that trains to chance for every mask.
"""

import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from olmo_core.data.document_chunk_landmark import reserved_ids

SCRIPT = Path(__file__).resolve().parents[3] / "recipes" / "train" / "run_guards.py"


@pytest.fixture(scope="module")
def guards():
    spec = importlib.util.spec_from_file_location("run_guards", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- token budget --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "max_tokens, rows, seq_len",
    [
        (4_000_000, 128, 65536),  # ds64 soft arm: 8.4M tokens/step -> 0 steps
        (8_000_000, 128, 65536),  # ...and 8M was the second byte-identical 1-step point
        (16_777_215, 128, 65536),  # one token short of two steps
    ],
)
def test_budget_below_two_steps_is_refused(guards, max_tokens, rows, seq_len):
    with pytest.raises(ValueError, match="ONE step"):
        guards.check_token_budget(max_tokens, rows, seq_len)


@pytest.mark.parametrize(
    "max_tokens, rows, seq_len, steps",
    [
        (4_000_000, 8, 65536, 7),  # dense arm of the same ladder maps fine
        (8_000_000, 8, 65536, 15),
        (16_777_216, 128, 65536, 2),  # exactly two steps
    ],
)
def test_budget_of_two_or_more_steps_passes(guards, max_tokens, rows, seq_len, steps):
    assert guards.check_token_budget(max_tokens, rows, seq_len) == steps


@pytest.mark.parametrize("max_tokens", [0, None])
def test_no_budget_always_passes(guards, max_tokens):
    assert guards.check_token_budget(max_tokens, 128, 65536) is None


# ---- run identity / silent auto-resume ----------------------------------------------------------


def _fake_step_checkpoint(save_folder: Path, step: int) -> None:
    """The trainer treats ``stepN/`` with a ``.metadata`` file as a resumable checkpoint."""
    d = save_folder / f"step{step}"
    d.mkdir(parents=True)
    (d / ".metadata").write_bytes(b"")


SLURM_A = {"SLURM_JOB_ID": "1001"}
SLURM_B = {"SLURM_JOB_ID": "2002"}


def test_launch_id_prefers_beaker_over_slurm(guards):
    assert (
        guards.launch_id({"BEAKER_EXPERIMENT_ID": "01ABC", "SLURM_JOB_ID": "7"}) == "beaker:01ABC"
    )
    assert guards.launch_id({"SLURM_JOB_ID": "7"}) == "slurm:7"
    assert guards.launch_id({}) is None


def test_fresh_folder_passes_and_records_identity(guards, tmp_path):
    save = tmp_path / "run"
    ident = guards.check_run_identity(str(save), "r1", allow_resume=False, env=SLURM_A)
    guards.write_run_identity(str(save), ident)
    rec = json.loads((save / guards.RUN_IDENTITY_FNAME).read_text())
    assert rec["run_name"] == "r1" and rec["launch_id"] == "slurm:1001"


def test_folder_without_step_checkpoints_passes_even_if_owned_by_another_run(guards, tmp_path):
    save = tmp_path / "run"
    guards.write_run_identity(
        str(save), guards.check_run_identity(str(save), "old", allow_resume=False, env=SLURM_A)
    )
    (save / "model_and_optim").mkdir()  # the model-only export is NOT a resume point
    (save / "model_and_optim" / ".metadata").write_bytes(b"")
    guards.check_run_identity(str(save), "new", allow_resume=False, env=SLURM_B)


def test_preemption_restart_of_same_launch_resumes(guards, tmp_path):
    save = tmp_path / "run"
    first = guards.check_run_identity(str(save), "r1", allow_resume=False, env=SLURM_A)
    guards.write_run_identity(str(save), first)
    _fake_step_checkpoint(save, 250)
    again = guards.check_run_identity(str(save), "r1", allow_resume=False, env=SLURM_A)
    guards.write_run_identity(str(save), again)
    assert "previous" not in json.loads((save / guards.RUN_IDENTITY_FNAME).read_text())


def test_beaker_retry_keeps_experiment_id_and_resumes(guards, tmp_path):
    save = tmp_path / "run"
    env1 = {"BEAKER_EXPERIMENT_ID": "01EXP", "BEAKER_JOB_ID": "job-1"}
    env2 = {"BEAKER_EXPERIMENT_ID": "01EXP", "BEAKER_JOB_ID": "job-2"}  # the preemption retry
    guards.write_run_identity(
        str(save), guards.check_run_identity(str(save), "r1", allow_resume=False, env=env1)
    )
    _fake_step_checkpoint(save, 100)
    guards.check_run_identity(str(save), "r1", allow_resume=False, env=env2)


@pytest.mark.parametrize(
    "run_name, env",
    [
        ("r1", SLURM_B),  # same name, NEW launch: the crash-relaunch trap
        ("r2", SLURM_A),  # another run's folder
        ("r1", {}),  # interactive torchrun: no launch id, cannot prove it is a restart
    ],
)
def test_new_launch_into_checkpointed_folder_is_refused(guards, tmp_path, run_name, env):
    save = tmp_path / "run"
    guards.write_run_identity(
        str(save), guards.check_run_identity(str(save), "r1", allow_resume=False, env=SLURM_A)
    )
    _fake_step_checkpoint(save, 250)
    with pytest.raises(RuntimeError, match="SILENTLY RESUME") as err:
        guards.check_run_identity(str(save), run_name, allow_resume=False, env=env)
    assert "step250" in str(err.value) and "--resume" in str(err.value)


def test_checkpoints_without_a_marker_are_refused(guards, tmp_path):
    save = tmp_path / "run"
    _fake_step_checkpoint(save, 500)
    with pytest.raises(RuntimeError, match="no run_identity.json"):
        guards.check_run_identity(str(save), "r1", allow_resume=False, env=SLURM_A)


def test_explicit_resume_passes_and_keeps_a_trail(guards, tmp_path):
    save = tmp_path / "run"
    guards.write_run_identity(
        str(save), guards.check_run_identity(str(save), "r1", allow_resume=False, env=SLURM_A)
    )
    _fake_step_checkpoint(save, 250)
    ident = guards.check_run_identity(str(save), "r1", allow_resume=True, env=SLURM_B)
    guards.write_run_identity(str(save), ident)
    rec = json.loads((save / guards.RUN_IDENTITY_FNAME).read_text())
    assert rec["launch_id"] == "slurm:2002"
    assert [p["launch_id"] for p in rec["previous"]] == ["slurm:1001"]
    # ...and the resumed launch's own preemption restart now passes without the flag.
    guards.check_run_identity(str(save), "r1", allow_resume=False, env=SLURM_B)


# ---- marker embeddings ---------------------------------------------------------------------------


def _embedding(family: str, markers, *, repaired: bool, dim: int = 16) -> torch.Tensor:
    """Trained rows at norm ~1.4; markers either one shared small vector (raw) or real-ish rows."""
    ids = reserved_ids(family)
    torch.manual_seed(0)
    emb = torch.randn(ids.real_vocab_size + 256, dim)
    emb = emb * (1.4 / emb.norm(dim=-1, keepdim=True))
    shared = torch.randn(dim) * 0.02
    for name in markers:
        tid = getattr(ids, name)
        emb[tid] = emb[100 + tid % 1000].clone() if repaired else shared
    return emb


MARKERS = ["doc_start", "doc_end", "landmark"]


def test_unrepaired_markers_fail_with_a_pointer_to_the_fix(guards):
    emb = _embedding("qwen3_5", MARKERS, repaired=False)
    with pytest.raises(RuntimeError, match="fix_marker_embeddings.py"):
        guards.check_marker_embeddings(emb, "qwen3_5", MARKERS, base_checkpoint="/b")


def test_repaired_markers_pass(guards):
    emb = _embedding("qwen3_5", MARKERS, repaired=True)
    report = guards.check_marker_embeddings(emb, "qwen3_5", MARKERS)
    assert report["audit_pass"]


def test_low_norm_alone_fails(guards):
    """Distinguishable markers at ~1/3.6 of a real token's norm still poison training."""
    emb = _embedding("qwen3_5", MARKERS, repaired=True)
    ids = reserved_ids("qwen3_5")
    for name in MARKERS:
        emb[getattr(ids, name)] *= 1 / 3.6
    with pytest.raises(RuntimeError, match="NOT repaired"):
        guards.check_marker_embeddings(emb, "qwen3_5", MARKERS)


def test_audit_base_checkpoint_never_raises(guards):
    bad = guards.audit_base_checkpoint(
        "/b",
        "qwen3_5",
        MARKERS,
        load_embedding=lambda _p: _embedding("qwen3_5", MARKERS, repaired=False),
    )
    assert not bad["ok"] and "NOT repaired" in bad["error"]

    def _boom(_p):
        raise FileNotFoundError("no such checkpoint")

    missing = guards.audit_base_checkpoint("/b", "qwen3_5", MARKERS, load_embedding=_boom)
    assert not missing["ok"] and "FileNotFoundError" in missing["error"]
    good = guards.audit_base_checkpoint(
        "/b",
        "qwen3_5",
        MARKERS,
        load_embedding=lambda _p: _embedding("qwen3_5", MARKERS, repaired=True),
    )
    assert good["ok"] and good["report"]["audit_pass"]


# ---- stack dumps ---------------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="no SIGUSR1")
def test_sigusr1_dumps_stacks_and_keeps_running(guards):
    code = (
        "import importlib.util, os, signal, sys\n"
        f"spec = importlib.util.spec_from_file_location('g', {str(SCRIPT)!r})\n"
        "g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)\n"
        "assert g.install_stack_dump_handler()\n"
        "def where_am_i():\n"
        "    os.kill(os.getpid(), signal.SIGUSR1)\n"
        "where_am_i()\n"
        "print('still alive', flush=True)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=os.environ
    )
    assert out.returncode == 0, out.stderr
    assert "still alive" in out.stdout
    assert "where_am_i" in out.stderr
