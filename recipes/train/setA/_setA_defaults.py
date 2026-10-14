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
* ``SETA_VARIANT``    -- a research variant from ``recipes/train/variants.py`` (default ``dense``).
* ``SETA_VARIANT_OPTS`` -- its options, ``k=v,...`` (``;``-separated if a value holds commas).
* ``SETA_MODEL``      -- the model (default ``qwen35-4b``, the dense Qwen3.5-4B recipe); see
  :data:`MODELS`. ``qwen3-4b`` is the full-attention Qwen3-4B-Base arm of the length-generalization
  experiment. It changes the shards dir, the default base, the window of its one arm (``32k``) and
  the run-name prefix.
* ``SETA_SMOKE_STEPS`` -- a smoke run: stop after this many steps (``hard_stop``, so the LR schedule
  is the full run's), log every step and add a GPU-memory monitor. Unset for real runs.

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
ENV_VARIANT = "SETA_VARIANT"
ENV_VARIANT_OPTS = "SETA_VARIANT_OPTS"
DEFAULT_VARIANT = "dense"
ENV_MODEL = "SETA_MODEL"
ENV_SMOKE_STEPS = "SETA_SMOKE_STEPS"
#: Optional overrides of the model's shard dir and of the 32k arm's packing window, for a re-rendered
#: data set cut to the baseline's examples (e.g. hashed doc ids: ``shards_qwen35_q35le32k`` at 65,536).
ENV_SHARDS_DIR = "SETA_SHARDS_DIR"
ENV_WINDOW = "SETA_WINDOW"
#: ``1``: let a relaunch into an existing save folder replace stale step dirs (a preemption mid-save
#: leaves a partial ``stepN``; the auto-resume from an earlier step then dies re-saving ``stepN``).
ENV_SAVE_OVERWRITE = "SETA_SAVE_OVERWRITE"

#: Qwen3-4B-Base converted to olmo-core (2026-07-23, HF-parity validated) with its marker rows
#: repaired from trained delimiter donors (``fix_marker_embeddings.py``); audited by
#: ``audit_qwen3_base.py``.
QWEN3_4B_BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/bases/"
    "qwen3-4b-base-trainedmark/model_and_optim"
)

DEFAULT_MODEL = "qwen35-4b"
#: Per-model recipe constants. Everything not listed here is the dense Qwen3.5-4B recipe.
#:
#: * ``shards_dir`` -- under the set root. ``shards_qwen3_q35le32k`` holds the setA rows
#:   re-tokenized with Qwen3 (``build_qwen3_setA_shards.py``) and cut to exactly the examples the
#:   Qwen3.5 32k arm kept (Qwen3.5 length <= 32,768).
#: * ``windows`` -- the packing window of each arm the model runs. For ``qwen3-4b`` the arm is still
#:   called ``32k`` (it is the 32k arm's example set), but Qwen3 tokenizes the same text longer, so
#:   the smallest power-of-two window that holds its longest kept example is 65,536.
MODELS = {
    "qwen35-4b": dict(
        run_prefix="q35-4b",
        shards_dir=SHARDS_DIR,
        base_checkpoint=DEFAULT_BASE_CHECKPOINT,
        marker_set="qwen3_5",
        windows={"32k": 32_768, "256k": 262_144, "256k-2node": 262_144},
    ),
    "qwen3-4b": dict(
        run_prefix="q3-4b",
        shards_dir="shards_qwen3_q35le32k",
        base_checkpoint=QWEN3_4B_BASE_CHECKPOINT,
        marker_set="qwen3",
        windows={"32k": 65_536},
    ),
}


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


def model_from_env() -> str:
    """:returns: ``SETA_MODEL``, or :data:`DEFAULT_MODEL`."""
    m = (os.environ.get(ENV_MODEL) or DEFAULT_MODEL).strip()
    if m not in MODELS:
        raise ValueError(f"{ENV_MODEL}={m!r}; expected one of {sorted(MODELS)}")
    return m


def base_checkpoint_from_env() -> str:
    """:returns: ``SETA_BASE_CKPT``, or the model's base (:data:`DEFAULT_BASE_CHECKPOINT` for dense
    Qwen3.5-4B)."""
    return os.environ.get(ENV_BASE_CKPT) or MODELS[model_from_env()]["base_checkpoint"]


def save_overwrite_from_env() -> bool:
    """:returns: whether ``SETA_SAVE_OVERWRITE=1``."""
    return os.environ.get(ENV_SAVE_OVERWRITE, "0") == "1"


def smoke_steps_from_env() -> int:
    """:returns: ``SETA_SMOKE_STEPS`` (0 = not a smoke run)."""
    return int(os.environ.get(ENV_SMOKE_STEPS) or 0)


def variant_from_env() -> str:
    """:returns: ``SETA_VARIANT``, or :data:`DEFAULT_VARIANT` (``full`` is spelled ``dense``)."""
    v = (os.environ.get(ENV_VARIANT) or DEFAULT_VARIANT).strip()
    return DEFAULT_VARIANT if v == "full" else v


def variant_opts_from_env() -> str:
    """:returns: ``SETA_VARIANT_OPTS`` (``""`` when unset)."""
    return os.environ.get(ENV_VARIANT_OPTS, "")


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


def shards_dir_from_env(model: str) -> str:
    """:returns: ``SETA_SHARDS_DIR``, else the model's ``shards_dir``."""
    return os.environ.get(ENV_SHARDS_DIR) or MODELS[model]["shards_dir"]


def window_from_env() -> int:
    """:returns: ``SETA_WINDOW`` (the 32k arm's packing window override; 0 = the model's own)."""
    return int(os.environ.get(ENV_WINDOW) or 0)
