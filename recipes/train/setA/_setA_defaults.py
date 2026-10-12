"""
Defaults of the setA SFT recipe, and their env-var overrides.

Import-free on purpose (stdlib only), so ``launch_setA_sft.py`` can read the defaults on a login node
without importing torch. ``_qwen35_setA_common.py`` resolves the same values on the Beaker node, where
the launcher has exported the overrides with ``gantry --env``:

* ``SETA_TASKS``      -- comma-separated task list (default: the 11 setA tasks). One task gives a
  single-task debugging run, e.g. ``SETA_TASKS=nq``.
* ``SETA_DATA_ROOT``  -- the set root (default :data:`DEFAULT_SET_ROOT`); shards are read from
  ``<root>/shards_qwen35_256k/<task>/``.
* ``SETA_BASE_CKPT``  -- the ``model_and_optim`` dir loaded weights-only
  (default :data:`DEFAULT_BASE_CHECKPOINT`).

The training scripts are driven by the olmo-core CLI (``<script> train <run-name> <cluster>``), which
has no room for extra argparse flags, hence env vars rather than arguments.
"""

import hashlib
import os
from typing import List

DEFAULT_SET_ROOT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_sft_sets/setA_max20_evaliid"
)
SHARDS_DIR = "shards_qwen35_256k"

#: The 11 setA tasks (textgroups dropped: shortest-document shortcut, see build_ctc_sft.py).
DEFAULT_TASKS: List[str] = [
    "nq",
    "hotpotqa",
    "qdmatch_nq",
    "outlier",
    "oolong",
    "contradiction",
    "xabsence",
    "reorder",
    "rerank",
    "strmatch",
    "grouping",
]

#: Dense 256k CPT base, loaded weights-only -- the base of amandab's dense 256k SFT runs.
DEFAULT_BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/q35-4b-dense-256k-fix/step2385/model_and_optim"
)

ENV_TASKS = "SETA_TASKS"
ENV_DATA_ROOT = "SETA_DATA_ROOT"
ENV_BASE_CKPT = "SETA_BASE_CKPT"


def parse_tasks(spec: str) -> List[str]:
    """
    Parse a comma-separated task list.

    :param spec: e.g. ``"nq,hotpotqa"``; empty means the default set.

    :returns: The task names, in the given order, de-duplicated.
    """
    tasks: List[str] = []
    for t in spec.split(","):
        t = t.strip()
        if t and t not in tasks:
            tasks.append(t)
    return tasks or list(DEFAULT_TASKS)


def tasks_from_env() -> List[str]:
    """:returns: ``SETA_TASKS`` parsed, or :data:`DEFAULT_TASKS` when it is unset/empty."""
    return parse_tasks(os.environ.get(ENV_TASKS, ""))


def set_root_from_env() -> str:
    """:returns: ``SETA_DATA_ROOT``, or :data:`DEFAULT_SET_ROOT`."""
    return os.environ.get(ENV_DATA_ROOT) or DEFAULT_SET_ROOT


def base_checkpoint_from_env() -> str:
    """:returns: ``SETA_BASE_CKPT``, or :data:`DEFAULT_BASE_CHECKPOINT`."""
    return os.environ.get(ENV_BASE_CKPT) or DEFAULT_BASE_CHECKPOINT


def task_tag(tasks: List[str]) -> str:
    """
    Run-name fragment for a non-default task set.

    :param tasks: The task list.

    :returns: ``""`` for the default 11-task set (in any order); otherwise the task names joined with
        ``_`` for up to three tasks (``"nq"``, ``"nq_hotpotqa"``), or ``"<k>tasks-<hash6>"`` for more.
    """
    if sorted(tasks) == sorted(DEFAULT_TASKS):
        return ""
    if len(tasks) <= 3:
        return "_".join(tasks)
    digest = hashlib.sha1(",".join(sorted(tasks)).encode()).hexdigest()[:6]
    return f"{len(tasks)}tasks-{digest}"
