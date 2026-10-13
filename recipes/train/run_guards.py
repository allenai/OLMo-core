"""
Start-of-run guards for the training entry points, each against a trap that has already cost runs.

* :func:`install_stack_dump_handler` -- ``kill -USR1 <pid>`` prints every thread's Python stack
  into the job log without killing the rank, so a hung job can be diagnosed in place (py-spy is not
  installed on the cluster by default; see ``recipes/env/local_env.sh``).
* :func:`check_token_budget` -- ``--max-tokens`` below two optimizer steps' worth of tokens silently
  gives a 1-step run (``max_duration`` is satisfied after step 1). Those fake points landed far left
  of the dense curve and scored the best matched-FLOP deltas of the ds64 ladder.
* :func:`check_run_identity` / :func:`write_run_identity` -- the trainer auto-resumes from any step
  checkpoint already in the save folder. A NEW launch into a reused folder then trains only the
  remaining steps from whatever produced that checkpoint (wrong step counts, wrong wall-clock,
  possibly a different base or config). A small ``run_identity.json`` marker records which launch
  owns the folder; a preemption restart of the SAME launch (same Beaker experiment / same slurm job
  id) keeps auto-resuming, anything else must pass an explicit resume flag.
* :func:`check_marker_embeddings` -- Qwen never trains the reserved marker rows the
  document-chunked / landmark data is built on: they are bit-identical (cos 1.0) and far below a
  real token's norm. Marker-dense runs then train to chance for every mask, which reads as a
  modeling result. Reuses the audit of ``recipes/data/fix_marker_embeddings.py``.

Stdlib-only at import time: olmo_core / torch are imported inside the functions that need them.
"""

import datetime
import json
import os
import socket
import sys
from typing import Callable, Dict, List, Mapping, Optional

#: Fewest optimizer steps a ``--max-tokens`` budget may resolve to.
MIN_OPTIMIZER_STEPS = 2

#: File in the save folder that records which launch owns it.
RUN_IDENTITY_FNAME = "run_identity.json"

_FIX_SCRIPT = "recipes/data/fix_marker_embeddings.py"


# ------------------------------------------------------------------------------------------------
# Stack dumps for hung jobs
# ------------------------------------------------------------------------------------------------


def install_stack_dump_handler() -> bool:
    """
    Make ``kill -USR1 <pid>`` dump every thread's Python stack to stderr (the job log) and keep
    running. Safe to call more than once. Does not touch any ``faulthandler.dump_traceback_later``
    watchdog a script already set.

    :returns: ``True`` if the handler was registered, ``False`` where SIGUSR1 does not exist.
    """
    import faulthandler
    import signal

    sig = getattr(signal, "SIGUSR1", None)
    if sig is None:  # pragma: no cover (Windows)
        return False
    faulthandler.register(sig, all_threads=True)
    return True


# ------------------------------------------------------------------------------------------------
# Token budget -> optimizer steps
# ------------------------------------------------------------------------------------------------


def check_token_budget(
    max_tokens: int, global_batch_instances: int, seq_len: int, *, flag: str = "--max-tokens"
) -> Optional[int]:
    """
    Refuse a token budget that cannot buy :data:`MIN_OPTIMIZER_STEPS` optimizer steps.

    :param max_tokens: The budget; ``0`` / ``None`` means "no budget" and always passes.
    :param global_batch_instances: Rows (instances) per optimizer step across all ranks.
    :param seq_len: Tokens per row.
    :param flag: The CLI flag name, for the error message.

    :returns: The number of optimizer steps the budget resolves to (``None`` without a budget).

    :raises ValueError: If the budget resolves to fewer than :data:`MIN_OPTIMIZER_STEPS` steps.
    """
    if not max_tokens:
        return None
    tokens_per_step = int(global_batch_instances) * int(seq_len)
    steps = int(max_tokens) // tokens_per_step
    if steps < MIN_OPTIMIZER_STEPS:
        raise ValueError(
            f"{flag} {max_tokens:,} buys {steps} optimizer step(s): one step is "
            f"{global_batch_instances} rows x {seq_len} tokens = {tokens_per_step:,} tokens, and "
            f"the run needs at least {MIN_OPTIMIZER_STEPS * tokens_per_step:,}. A budget below that "
            "silently trains for ONE step and looks like a real data-scaling point. Raise "
            f"{flag}, or cut --global-batch as well."
        )
    return steps


# ------------------------------------------------------------------------------------------------
# Run identity (silent auto-resume)
# ------------------------------------------------------------------------------------------------


def launch_id(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """
    The id of the launch this process belongs to, stable across a preemption restart of that launch.

    Beaker retries a preempted job inside the same experiment (``BEAKER_EXPERIMENT_ID`` stays, the
    job id changes); a requeued slurm job keeps ``SLURM_JOB_ID``. A bare ``torchrun`` has neither.

    :param env: Environment to read (default ``os.environ``).

    :returns: ``"beaker:<experiment id>"``, ``"slurm:<job id>"``, or ``None``.
    """
    env = os.environ if env is None else env
    if env.get("BEAKER_EXPERIMENT_ID"):
        return f"beaker:{env['BEAKER_EXPERIMENT_ID']}"
    if env.get("SLURM_JOB_ID"):
        return f"slurm:{env['SLURM_JOB_ID']}"
    return None


def find_step_checkpoints(save_folder: str) -> List[str]:
    """
    The step checkpoints the trainer would auto-resume from (same detection as the trainer).

    :param save_folder: The run's save folder.

    :returns: Checkpoint directory paths, sorted by step; empty if the folder does not exist.
    """
    if not os.path.isdir(save_folder):
        return []
    from olmo_core.train.checkpoint import Checkpointer

    try:
        return [p for _, p in sorted(Checkpointer.find_checkpoints(save_folder))]
    except FileNotFoundError:
        return []


def read_run_identity(save_folder: str) -> Optional[dict]:
    """
    :param save_folder: The run's save folder.

    :returns: The recorded identity, or ``None`` if there is no (readable) marker.
    """
    path = os.path.join(save_folder, RUN_IDENTITY_FNAME)
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def check_run_identity(
    save_folder: str,
    run_name: str,
    *,
    allow_resume: bool,
    env: Optional[Mapping[str, str]] = None,
    resume_flag: str = "--resume",
) -> dict:
    """
    Refuse to start a NEW launch in a save folder that holds step checkpoints of another launch.

    Passes when the folder has no step checkpoint, when ``allow_resume`` is set, or when the marker
    names this run and this launch (a preemption restart). Fails when checkpoints exist and the
    marker is missing, names another run, or names another launch (or this process has no launch
    id, e.g. an interactive ``torchrun``).

    :param save_folder: The run's save folder (a local path).
    :param run_name: This run's name.
    :param allow_resume: The explicit opt-in to continue whatever checkpoint is there.
    :param env: Environment to read the launch id from (default ``os.environ``).
    :param resume_flag: The CLI flag name, for the error message.

    :returns: The identity to record for this launch (pass it to :func:`write_run_identity`).

    :raises RuntimeError: If a new launch would silently resume another launch's checkpoint.
    """
    identity = {
        "run_name": run_name,
        "launch_id": launch_id(env),
        "host": socket.gethostname(),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": list(sys.argv),
    }
    ckpts = find_step_checkpoints(save_folder)
    if not ckpts or allow_resume:
        return identity
    prev = read_run_identity(save_folder)
    same_launch = (
        prev is not None
        and prev.get("run_name") == run_name
        and identity["launch_id"] is not None
        and prev.get("launch_id") == identity["launch_id"]
    )
    if same_launch:
        return identity
    owner = (
        "no run_identity.json (written by an older trainer, or by hand)"
        if prev is None
        else f"run_name={prev.get('run_name')!r} launch_id={prev.get('launch_id')!r} "
        f"started={prev.get('started')!r}"
    )
    raise RuntimeError(
        f"save folder {save_folder} already holds {len(ckpts)} step checkpoint(s) (latest "
        f"{os.path.basename(ckpts[-1])}) owned by {owner}; this launch is run_name={run_name!r} "
        f"launch_id={identity['launch_id']!r}. The trainer would SILENTLY RESUME from it: the step "
        "counts, wall-clock and possibly the base/config would all come from the old run. Use a "
        f"fresh --run-name / --save-folder, delete the old checkpoints, or pass {resume_flag} to "
        "continue that checkpoint on purpose."
    )


def write_run_identity(save_folder: str, identity: dict) -> None:
    """
    Record ``identity`` as the owner of ``save_folder`` (atomic replace). Earlier owners are kept
    under ``"previous"`` so an explicit resume leaves a trail. Call from ONE rank.

    :param save_folder: The run's save folder.
    :param identity: From :func:`check_run_identity`.
    """
    os.makedirs(save_folder, exist_ok=True)
    prev = read_run_identity(save_folder)
    record = dict(identity)
    if prev is not None:
        same = prev.get("run_name") == identity.get("run_name") and prev.get(
            "launch_id"
        ) == identity.get("launch_id")
        history = list(prev.get("previous", []))
        if not same:
            history.append({k: v for k, v in prev.items() if k != "previous"})
        else:
            record["started"] = prev.get("started", record.get("started"))
        if history:
            record["previous"] = history
    path = os.path.join(save_folder, RUN_IDENTITY_FNAME)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------
# Marker embeddings
# ------------------------------------------------------------------------------------------------


def _fix_marker_module():
    """Import ``recipes/data/fix_marker_embeddings.py`` (it is a script, not a package module)."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    path = os.path.join(path, "fix_marker_embeddings.py")
    spec = importlib.util.spec_from_file_location("fix_marker_embeddings", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_marker_embeddings(
    emb,
    family: str,
    markers: List[str],
    *,
    base_checkpoint: str = "<base>",
) -> dict:
    """
    Fail if the marker rows of an embedding matrix are degenerate (the audit of
    ``fix_marker_embeddings.py``: every pair ``|cos| < COS_MAX`` AND every marker norm within
    ``(NORM_RATIO_MIN, NORM_RATIO_MAX)`` x the median trained-row norm).

    :param emb: The ``[vocab, d_model]`` input-embedding matrix.
    :param family: Key of :data:`olmo_core.data.document_chunk_landmark.RESERVED_IDS`.
    :param markers: Marker names to check (``doc_start``, ``doc_end``, ``landmark``, ...).
    :param base_checkpoint: For the error message.

    :returns: The audit report.

    :raises RuntimeError: If either gate fails.
    """
    from olmo_core.data.document_chunk_landmark import RESERVED_IDS

    fme = _fix_marker_module()
    ids = RESERVED_IDS[family]
    report = fme.audit(emb, ids, fme._marker_ids(ids, markers))
    if not report["audit_pass"]:
        ratios = {k: round(v, 3) for k, v in report["marker_norm_ratios"].items()}
        cosines = {k: round(v, 4) for k, v in report["pairwise_cosine"].items()}
        raise RuntimeError(
            f"marker embeddings of {base_checkpoint} are NOT repaired (family={family}): "
            f"pairwise cos {cosines} (gate |cos| < {fme.COS_MAX}), norm / median trained row "
            f"{ratios} (gate {fme.NORM_RATIO_MIN}..{fme.NORM_RATIO_MAX}). Marker-dense training "
            "on this base flatlines at chance for EVERY mask, including plain causal. Run "
            f"{_FIX_SCRIPT} on the base and train from the repaired copy (see CLAUDE.md, "
            "records/document-chunked-marker-embeddings.md)."
        )
    return report


def audit_base_checkpoint(
    base_checkpoint: str,
    family: str,
    markers: List[str],
    *,
    load_embedding: Optional[Callable[[str], object]] = None,
) -> Dict[str, object]:
    """
    Read the input embedding out of a distcp base and run :func:`check_marker_embeddings`. Never
    raises: the result is ``{"ok": bool, "error": str | None, "report": dict | None}`` so one rank
    can audit and broadcast the verdict to the others.

    :param base_checkpoint: The ``model_and_optim`` distcp directory.
    :param family: Key of ``RESERVED_IDS``.
    :param markers: Marker names to check.
    :param load_embedding: Override of the loader (tests).
    """
    try:
        if load_embedding is None:
            fme = _fix_marker_module()
            emb = next(iter(fme.load_keys(base_checkpoint, [fme.EMB_KEY]))).float()
        else:
            emb = load_embedding(base_checkpoint)
        report = check_marker_embeddings(emb, family, markers, base_checkpoint=base_checkpoint)
        return {"ok": True, "error": None, "report": report}
    except Exception as exc:  # noqa: BLE001 -- surfaced verbatim on every rank
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "report": None}
