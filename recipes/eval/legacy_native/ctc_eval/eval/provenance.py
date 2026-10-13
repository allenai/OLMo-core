"""
Provenance for the native multirung eval JSONs (``eval_lc_native.py``,
``eval_lc_native_docchunk_ladder.py``), so a result file says how many examples each number came
from, which job and commit produced it, and which checkpoint it scored.

Everything goes INSIDE the ``_meta`` block, never at the top level: results-hub's ingester treats
any top-level dict of numbers as another task's scores, and that would also flip a flat ladder file
into its per-task shape. ``ops result add --json FILE`` reads these ``_meta`` keys
(``ops/opslib/results.py``), so a file written with them can be recorded unattended.

Each eval pass is its own run in the ops registry: ``_meta.eval_run_id`` is ``$OPS_EVAL_RUN_ID``
(set by the Beaker eval launcher / ``recipes/env/ops_job.sh``) or else
``<train_run>__ev-<suite>-<variant>-<YYYYmmddHHMM>``; ``_meta.train_run`` is the checkpoint's run.

With ``OPS_RECORD_RESULTS=1`` the driver also registers the pass and appends the rows to ``ops/state/results.jsonl``
itself. Only useful where the canonical registry is reachable (Berkeley slurm via NFS); a Beaker
container cannot reach it, so record those from the login node after pulling the JSON.
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]  # .../recipes/eval/legacy_native/ctc_eval/eval/
COT_MODE_LABELS = {"none": "no-cot"}  # other modes are not mapped: label them by hand


def eval_job_id():
    """Beaker workload/experiment/job id, else the slurm job id, else ``None``."""
    for var in ("BEAKER_WORKLOAD_ID", "BEAKER_EXPERIMENT_ID", "BEAKER_JOB_ID", "SLURM_JOB_ID"):
        if os.environ.get(var):
            return os.environ[var]
    return None


def git_commit():
    """HEAD of the checkout this driver runs from (gantry clones the launched commit)."""
    try:
        res = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return (res.stdout.strip() or None) if res.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def train_run_from_ckpt(path):
    """Run name from a checkpoint path (drops trailing ``stepN[_yarnK]`` / ``model_and_optim`` / ``hf``)."""
    parts = [p for p in str(path).rstrip("/").split("/") if p]
    while parts and (re.fullmatch(r"step\d+(-unsharded|_yarn\d+)?", parts[-1]) or parts[-1] in ("model_and_optim", "hf")):
        parts.pop()
    return parts[-1] if parts else None


def eval_run_id(train_run, suite, variant):
    """``$OPS_EVAL_RUN_ID``, else ``<train_run>__ev-<suite>-<variant>-<YYYYmmddHHMM>`` (ops convention)."""
    if os.environ.get("OPS_EVAL_RUN_ID"):
        return os.environ["OPS_EVAL_RUN_ID"]
    slug = lambda s: re.sub(r"[^A-Za-z0-9.+]+", "-", str(s)).strip("-") or "x"  # noqa: E731
    return f"{train_run}__ev-{slug(suite)}-{slug(variant)}-{datetime.now().strftime('%Y%m%d%H%M')}"


def eval_meta(model_path, eval_sizes, cot_label=None, suite="ladder", variant="default"):
    """Keys to merge into a result file's ``_meta`` block.

    :param model_path: The checkpoint (step dir) that was scored.
    :param eval_sizes: ``{"<task>_<rung>": number of examples graded}``.
    :param cot_label: ``long-cot`` / ``short-cot`` / ``no-cot`` when known, else ``None``.
    :param suite: Tasks scored by this pass (goes into a synthesized eval run id).
    :param variant: How they were scored, e.g. ``docchunk-dense`` or ``yarn2`` (ditto).
    """
    train_run = train_run_from_ckpt(model_path)
    meta = {
        "eval_run_id": eval_run_id(train_run, suite, variant),
        "train_run": train_run,
        "eval_size": dict(eval_sizes),
        "eval_job_id": eval_job_id(),
        "git_commit": git_commit(),
        "ckpt_path": model_path,
    }
    if cot_label:
        meta["cot_label"] = cot_label
    return meta


def maybe_record_results(out_path):
    """With ``OPS_RECORD_RESULTS=1``: register the eval pass (if new), then
    ``ops result add --json out_path --check``. Never raises."""
    if os.environ.get("OPS_RECORD_RESULTS") != "1":
        return None
    try:
        if str(REPO_ROOT / "ops") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "ops"))
        import json

        from opslib import safe

        meta = json.loads(Path(out_path).read_text()).get("_meta") or {}
        if meta.get("eval_run_id") and meta.get("train_run"):
            safe.register_eval_pass(
                meta["eval_run_id"],
                meta["train_run"],
                ckpt_path=meta.get("ckpt_path"),
                git_commit=meta.get("git_commit"),
                notes=f"result {out_path}",
            )
        return safe.record_results(out_path, check=True)
    except Exception as e:  # noqa: BLE001 - the result file is already written; never fail the eval
        print(f"[ops] WARNING: result recording failed ({type(e).__name__}: {e})", flush=True)
        return None
