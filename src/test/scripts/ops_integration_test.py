"""
Tests for the repo's hooks into the ``ops`` run registry and results log (``ops/``, stdlib-only).

Every hook writes to ``$OPS_STATE_DIR`` (a tmp dir here) and must never fail or block the launch,
eval or job it rides on: a registry problem prints one ``[ops] WARNING`` line and returns.

- ``recipes/train/research_trainer/beaker_ctc_suite.py``: ``register_run`` after submission.
- ``recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py``: ``register_eval`` per eval job.
- ``recipes/eval/legacy_native/ctc_eval/eval/provenance.py``: ``_meta`` provenance of result JSONs.
- ``recipes/env/ops_job.sh``: slurm payload wrapper (register, guard, final_state).
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
BEAKER_LAUNCHER = REPO / "recipes" / "train" / "research_trainer" / "beaker_ctc_suite.py"
EVAL_LAUNCHER = REPO / "recipes" / "eval" / "legacy_native" / "run_q4b_beaker_multirung_eval.py"
WRAPPER = REPO / "recipes" / "env" / "ops_job.sh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _runs(state: Path) -> list:
    path = state / "runs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("OPS_EXPERIMENT", raising=False)
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sess-1")
    sys.path.insert(0, str(REPO / "ops"))
    from opslib import registry

    # git_info shells out to git on the (NFS) checkout; the hooks pass the shipped commit anyway.
    monkeypatch.setattr(registry, "git_info", lambda _d: {"git_commit": None, "git_branch": None, "dirty": None})
    return tmp_path


@pytest.fixture(scope="module")
def beaker_launcher():
    return _load(BEAKER_LAUNCHER, "beaker_ctc_suite_under_test")


def _opts(**kw):
    base = dict(task="outlier", variant="chunked-mix", experiment=None, arm=None, expected_minutes=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_beaker_register_run_builds_row(beaker_launcher, state, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["beaker_ctc_suite.py", "--task", "outlier", "launch"])
    row = beaker_launcher.register_run(
        _opts(expected_minutes=240.0),
        run_name="ctc-outlier-cmix-4b-20261012T120000-0700",
        save_folder="/weka/x/ckpts/ctc-outlier-cmix-4b-20261012T120000-0700",
        wandb_group="ctc-suite-outlier",
        git_commit="deadbeef",
        git_branch="prasann/flopdrop_clean",
        dirty=False,
        job_id="01JOB",
    )
    assert row is not None
    (stored,) = _runs(state)
    assert stored == row
    assert stored["run_id"] == "ctc-outlier-cmix-4b-20261012T120000-0700"
    assert stored["experiment"] == "ctc_suite_outlier"  # slug of the wandb group
    assert (stored["kind"], stored["cluster"], stored["job_id"]) == ("train", "beaker", "01JOB")
    assert (stored["task"], stored["arm"]) == ("outlier", "chunked-mix")
    assert stored["ckpt_path"].endswith("/ckpts/ctc-outlier-cmix-4b-20261012T120000-0700")
    assert stored["git_commit"] == "deadbeef" and stored["git_branch"] == "prasann/flopdrop_clean"
    assert stored["wandb_url"].endswith("/groups/ctc-suite-outlier")
    assert stored["expected_minutes"] == 240.0
    assert stored["command"] == "beaker_ctc_suite.py --task outlier launch"
    assert stored["launched_by"] == "sess-1"
    assert stored["source"] == "launch"


def test_beaker_register_run_explicit_experiment_and_arm(beaker_launcher, state):
    row = beaker_launcher.register_run(
        _opts(experiment="ds64", arm="chk32"), run_name="r", save_folder="/s", wandb_group="g", job_id="J"
    )
    assert (row["experiment"], row["arm"]) == ("ds64", "chk32")


def test_beaker_register_run_never_raises(beaker_launcher, state, monkeypatch, capsys):
    # an invalid row (registry validation error) is reported, not raised
    row = beaker_launcher.register_run(
        _opts(expected_minutes="soon"), run_name="r", save_folder="/s", wandb_group="g", job_id="J"
    )
    assert row is None
    assert "[ops] WARNING" in capsys.readouterr().err
    assert _runs(state) == []

    # ops missing entirely
    def boom():
        raise RuntimeError("no ops here")

    monkeypatch.setattr(beaker_launcher, "_ops_safe", boom)
    assert beaker_launcher.register_run(_opts(), run_name="r", save_folder="/s", wandb_group="g") is None
    assert "[ops] WARNING" in capsys.readouterr().out


def test_beaker_launcher_new_flags(beaker_launcher, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--task", "nq", "--variant", "full", "--model-scale", "4b", "--run-name", "r",
         "--experiment", "flop_scaling", "--arm", "kv17", "--expected-minutes", "90", "dry_run"],
    )
    opts = beaker_launcher.parse_args()
    assert (opts.experiment, opts.arm, opts.expected_minutes) == ("flop_scaling", "kv17", 90.0)


def test_eval_launcher_register_eval(state, monkeypatch):
    mod = _load(EVAL_LAUNCHER, "run_q4b_beaker_multirung_eval_under_test")
    monkeypatch.setattr(sys, "argv", ["run_q4b_beaker_multirung_eval.py", "ds64-nq-dense-u16M", "ai2/neptune"])
    args = argparse.Namespace(run_name="ds64-nq-dense-u16M", experiment=None, ckpt="/weka/c/step9",
                              expected_minutes=None)
    lc = argparse.Namespace(name="ev-nq-ds64-nq-dense-u16M", git=argparse.Namespace(ref="cafe", branch="b"))
    assert mod.eval_pass_id(lc) == "ev-nq-ds64-nq-dense-u16M"  # natural name while unused
    row = mod.register_eval(args, lc, "nq", "01EV")
    assert row is not None
    assert (row["run_id"], row["kind"], row["parent_run"]) == ("ev-nq-ds64-nq-dense-u16M", "eval", "ds64-nq-dense-u16M")
    assert (row["experiment"], row["job_id"], row["git_commit"], row["ckpt_path"]) == ("ds64", "01EV", "cafe", "/weka/c/step9")
    assert mod.eval_pass_id(lc).startswith("ev-nq-ds64-nq-dense-u16M-")  # a second pass gets its own id
    # a broken launch config is reported, not raised
    assert mod.register_eval(args, object(), "nq", "01EV") is None


def test_eval_provenance_meta_and_recording(state, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(REPO / "recipes" / "eval" / "legacy_native"))
    from ctc_eval.eval import provenance

    monkeypatch.setenv("BEAKER_WORKLOAD_ID", "01EVJOB")
    monkeypatch.delenv("OPS_EVAL_RUN_ID", raising=False)
    meta = provenance.eval_meta("/w/ckpts/run-a/step10", {"nq_8k": 500}, cot_label="no-cot", suite="nq",
                                variant="docchunk-dense")
    # each eval pass is its own run: <train>__ev-<suite>-<variant>-<YYYYmmddHHMM> unless the launcher set one
    assert meta["train_run"] == "run-a"
    assert meta["eval_run_id"].startswith("run-a__ev-nq-docchunk-dense-")
    assert meta["eval_size"] == {"nq_8k": 500}
    assert (meta["eval_job_id"], meta["ckpt_path"], meta["cot_label"]) == ("01EVJOB", "/w/ckpts/run-a/step10", "no-cot")
    assert "git_commit" in meta

    out = tmp_path / "nq_multirung.json"
    out.write_text(json.dumps({"model_path": "/w/ckpts/run-a/step10", "nq_8k": 0.61, "_meta": meta}))
    monkeypatch.delenv("OPS_RECORD_RESULTS", raising=False)
    assert provenance.maybe_record_results(out) is None  # off unless asked
    monkeypatch.setenv("OPS_RECORD_RESULTS", "1")
    (row,) = provenance.maybe_record_results(out)
    assert (row["run_id"], row["train_run"]) == (meta["eval_run_id"], "run-a")
    assert (row["task"], row["rung"], row["eval_size"]) == ("nq", "8k", 500)
    assert (row["eval_job_id"], row["cot_label"]) == ("01EVJOB", "no-cot")
    (ev,) = _runs(state)  # the pass was registered before its rows were filed
    assert (ev["run_id"], ev["kind"], ev["parent_run"]) == (meta["eval_run_id"], "eval", "run-a")

    monkeypatch.setenv("OPS_EVAL_RUN_ID", "ev-nq-run-a")  # set by the Beaker eval launcher / ops_job.sh
    assert provenance.eval_meta("/w/ckpts/run-a", {})["eval_run_id"] == "ev-nq-run-a"


def _wrap(state: Path, *payload: str, **env):
    clean = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    clean.update({"OPS_STATE_DIR": str(state), "OPS_BIN": str(REPO / "ops" / "bin" / "ops"), **env})
    return subprocess.run(["bash", str(WRAPPER), *payload], env=clean, capture_output=True, text=True, timeout=120)


def test_ops_job_wrapper_success_and_failure(tmp_path):
    ok = _wrap(tmp_path, "bash", "-c", "echo payload-ran", OPS_RUN_ID="w-ok", OPS_EXPERIMENT="e",
               OPS_CKPT_PATH="/data/ckpts/w-ok", OPS_GUARD="0")
    assert ok.returncode == 0, ok.stderr
    assert "payload-ran" in ok.stdout
    bad = _wrap(tmp_path, "bash", "-c", "echo guarded; exit 3", OPS_RUN_ID="w-bad", OPS_EXPERIMENT="e",
                OPS_KIND="data")
    assert bad.returncode == 3, bad.stderr  # the payload's exit code passes through the guard
    assert "guarded" in bad.stdout
    rows = _runs(tmp_path)
    launches = {r["run_id"]: r for r in rows if r["event"] == "launch"}
    updates = {r["run_id"]: r for r in rows if r["event"] == "update"}
    assert launches["w-ok"]["cluster"] == "local" and launches["w-ok"]["ckpt_path"] == "/data/ckpts/w-ok"
    assert launches["w-bad"]["kind"] == "data" and launches["w-bad"]["source"] == "sbatch"
    assert updates["w-ok"]["final_state"] == "done"
    assert updates["w-bad"]["final_state"] == "failed"
    assert updates["w-bad"]["state_source"] == "ops_job.sh exit 3"


def test_ops_job_wrapper_eval_pass(tmp_path):
    res = _wrap(tmp_path, "bash", "-c", 'echo "id=$OPS_EVAL_RUN_ID"', OPS_KIND="eval", OPS_PARENT_RUN="train-z",
                OPS_TASK="contra", OPS_EVAL_VARIANT="yarn2", OPS_EXPERIMENT="e", OPS_GUARD="0")
    assert res.returncode == 0, res.stderr
    (launch, update) = _runs(tmp_path)
    assert launch["run_id"].startswith("train-z__ev-contra-yarn2-")
    assert (launch["kind"], launch["parent_run"]) == ("eval", "train-z")
    assert f"id={launch['run_id']}" in res.stdout  # the payload sees its own pass id
    assert update["final_state"] == "done"


def test_ops_job_wrapper_runs_payload_without_ops(tmp_path):
    res = _wrap(tmp_path, "bash", "-c", "echo still-ran", OPS_BIN="/nonexistent/ops", OPS_RUN_ID="w")
    assert res.returncode == 0
    assert "still-ran" in res.stdout
    assert "[ops] WARNING" in res.stderr
