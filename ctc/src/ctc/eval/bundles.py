"""
The eval bundle: which file grades which task at which rung.

A rung is a *token* budget, but a file on disk is a document count, and the two are related by a
per-task calibration that lives nowhere in the filename. ``contradiction`` at 8k is
``contradiction_eval_pubmed_both_n190_k3.jsonl``; ``scifact`` at 8k is ``beir_scifact_ladder_k22``.
Nobody can be expected to remember that, and every place it was retyped is a place it drifted --
so it is written down once, here, and ``--task nq --rungs all`` is enough to run.

**These are the reliable, per-example-corpus ladders.** Every row carries its own independently
sampled corpus, which is what makes 500 rows 500 independent measurements. The efficient
shared-corpus variants -- many queries over one corpus, so a prefill can be reused -- are a
*different* construction that measurably changes some tasks' scores, and they are built and named
separately rather than swapped in behind the same task name.

Two properties hold across every ladder here, and both are load-bearing:

* **A task's rungs grade the same questions.** Only the distractor documents change. The v1 ladders
  drew fresh questions per rung, so every rung-to-rung delta carried eval-set resampling noise on
  top of the length effect it was supposed to isolate; they are not reproduced.
* **Rung labels bound corpus size, not prompt length.** Measured contradiction prompts at the "4k"
  rung spanned 3,457 to 23,796 tokens. Sizing a decode budget from the label is how 354 of 500
  examples once got silently skipped and scored 0.0. :mod:`ctc.eval.runner` audits this; the labels
  here are names, not promises.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_ROOT",
    "BundleTask",
    "BUNDLE",
    "GROUPS",
    "bundle_root",
    "get",
    "names",
    "resolve",
]

#: The weka bundle every Beaker eval reads. ``_clean`` rather than the original
#: ``_eval_bundle_eval500_v2``: the contradiction rungs in that one were calibrated against a filler
#: pool that turned out to be 92-99% FEVER/wiki rather than PubMed, so every contradiction rung
#: overshot its label by ~1.8x. The clean bundle is the default and has been since 2026-07-29.
DEFAULT_ROOT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v2_clean"
)

#: Environment variable overriding :data:`DEFAULT_ROOT`, so a local run points at a staged copy
#: without editing anything.
ROOT_ENV = "CTC_EVAL_BUNDLE"


@dataclass(frozen=True)
class BundleTask:
    """
    One gradable ladder.

    :param name: What ``--task`` takes. Not always a registered spec name -- ``contra_fever`` and
        ``outlier_review`` are out-of-distribution *sources* graded by the in-distribution
        contradiction and outlier specs, and keeping them as separate ladder names is what lets a
        results table tell them apart.
    :param spec: The registered :class:`~ctc.format.registry.TaskSpec` that grades it.
    :param rungs: ``(label, path relative to the bundle root)``, ascending by context length.
    :param group: ``"main"`` for the five in-distribution tasks, ``"ood"`` for the held-out ones.
    :param note: Anything a reader of the numbers has to know.
    """

    name: str
    spec: str
    rungs: Tuple[Tuple[str, str], ...]
    group: str = "main"
    note: str = ""

    @property
    def labels(self) -> List[str]:
        """:returns: This task's rung labels, ascending."""
        return [label for label, _ in self.rungs]


#: The five in-distribution tasks plus the four out-of-distribution ladders. Paths are relative to
#: the bundle root and are the exact filenames the pre-migration driver resolved, so a number
#: produced here is comparable to the existing results grid rather than merely similar to it.
BUNDLE: Dict[str, BundleTask] = {
    "contradiction": BundleTask(
        name="contradiction",
        spec="contradiction",
        rungs=(
            ("2k", "contra/contradiction_eval_pubmed_both_n100_k3.jsonl"),
            ("8k", "contra/contradiction_eval_pubmed_both_n190_k3.jsonl"),
            ("16k", "contra/contradiction_eval_pubmed_both_n385_k3.jsonl"),
            ("32k", "contra/contradiction_eval_pubmed_both_n765_k3.jsonl"),
        ),
        note="PubMed claims. Gold indices are 1-based for this task and 0-based for every other.",
    ),
    "nq": BundleTask(
        name="nq",
        spec="retrieval",
        rungs=(
            ("3k", "nq/nq_validation_k20_600.jsonl"),
            ("8k", "nq/nq_validation_k50_600.jsonl"),
            ("16k", "nq/nq_validation_k100_600.jsonl"),
            ("32k", "nq/nq_validation_k200_600.jsonl"),
        ),
        note="Graded on retrieved document ids, not answer text.",
    ),
    "outlier": BundleTask(
        name="outlier",
        spec="outlier",
        rungs=(
            ("3k", "outlier/outlier_wiki100w_n22_k3_eval_600.jsonl"),
            ("8k", "outlier/outlier_wiki100w_n55_k3_eval_600.jsonl"),
            ("16k", "outlier/outlier_wiki100w_n110_k3_eval_600.jsonl"),
            ("32k", "outlier/outlier_wiki100w_n220_k3_eval_600.jsonl"),
        ),
    ),
    "rerank": BundleTask(
        name="rerank",
        spec="rerank",
        rungs=(
            ("3k", "rerank/msmarco_trainhn_eval_k20_500.jsonl"),
            ("8k", "rerank/msmarco_trainhn_eval_k50_500.jsonl"),
            ("16k", "rerank/msmarco_trainhn_eval_k100_500.jsonl"),
        ),
        note="No 32k rung: the CE-filtered hard-negative pool caps at 100 documents per query.",
    ),
    "oolong": BundleTask(
        name="oolong",
        spec="oolong",
        rungs=(
            ("8k", "oolong/oolong_test_synth_ctx8192_spliteval.jsonl"),
            ("16k", "oolong/oolong_test_synth_ctx16384_spliteval.jsonl"),
            ("32k", "oolong/oolong_test_synth_ctx32768_spliteval.jsonl"),
        ),
        note=(
            "Items are lines within one context block, not separate documents, so this is the one "
            "task whose chunk layout is line-based -- see the spec's extra['chunk_by']."
        ),
    ),
    "fiqa": BundleTask(
        name="fiqa",
        spec="retrieval",
        group="ood",
        rungs=(
            ("2k", "beir/beir_fiqa_ce_ladder_k10_648.jsonl"),
            ("4k", "beir/beir_fiqa_ce_ladder_k20_648.jsonl"),
            ("8k", "beir/beir_fiqa_ce_ladder_k40_648.jsonl"),
            ("16k", "beir/beir_fiqa_ce_ladder_k80_648.jsonl"),
        ),
        note="Held-out BEIR retrieval; graded by the same retrieval spec as nq.",
    ),
    "scifact": BundleTask(
        name="scifact",
        spec="retrieval",
        group="ood",
        rungs=(
            ("4k", "beir/beir_scifact_ladder_k11_299.jsonl"),
            ("8k", "beir/beir_scifact_ladder_k22_299.jsonl"),
            ("16k", "beir/beir_scifact_ladder_k44_299.jsonl"),
            ("32k", "beir/beir_scifact_ladder_k88_299.jsonl"),
        ),
        note=(
            "eval_size=299, below the 500 floor: quote the size and its error bar (about ±0.026 at "
            "f1 0.7) inline next to every number."
        ),
    ),
    "outlier_review": BundleTask(
        name="outlier_review",
        spec="outlier",
        group="ood",
        rungs=(
            ("3k", "outlier/outlier_review_matched_n30_k3_eval_600.jsonl"),
            ("8k", "outlier/outlier_review_matched_n75_k3_eval_600.jsonl"),
            ("16k", "outlier/outlier_review_matched_n150_k3_eval_600.jsonl"),
            ("32k", "outlier/outlier_review_matched_n300_k3_eval_600.jsonl"),
        ),
        note=(
            "Amazon-review passages instead of wiki100w. The `matched` build is the only one the "
            "grid has ever used -- it is difficulty-matched to the in-distribution outlier ladder, "
            "so the OOD gap is a source effect and not a difficulty effect."
        ),
    ),
    "contra_fever": BundleTask(
        name="contra_fever",
        spec="contradiction",
        group="ood",
        rungs=(
            ("2k", "contra/contradiction_eval_fever_plain_n100_k3.jsonl"),
            ("8k", "contra/contradiction_eval_fever_plain_n408_k3.jsonl"),
            ("16k", "contra/contradiction_eval_fever_plain_n820_k3.jsonl"),
            ("32k", "contra/contradiction_eval_fever_plain_n1642_k3.jsonl"),
        ),
        note=(
            "FEVER claims instead of PubMed. `plain` is the difficulty-matched build (no hard NEI "
            "pairs, no decoy support, no decoys) and is the only variant the grid uses."
        ),
    ),
}

#: Shorthand for ``--tasks``. ``main`` is the headline table; ``all`` adds the held-out ladders.
GROUPS: Dict[str, Tuple[str, ...]] = {
    "main": tuple(t.name for t in BUNDLE.values() if t.group == "main"),
    "ood": tuple(t.name for t in BUNDLE.values() if t.group == "ood"),
    "all": tuple(BUNDLE),
}


def bundle_root(explicit: Optional[str] = None) -> Path:
    """
    Resolve the bundle root: an explicit path, else ``$CTC_EVAL_BUNDLE``, else :data:`DEFAULT_ROOT`.

    :param explicit: A path from the command line, if given.

    :returns: The root directory. Not checked for existence here -- a launcher resolves this on the
        submitting host, where the weka mount does not exist.
    """
    return Path(explicit or os.environ.get(ROOT_ENV) or DEFAULT_ROOT)


def names(group: str = "all") -> List[str]:
    """
    :param group: A key of :data:`GROUPS`, or a comma list of task names.

    :returns: Task names.

    :raises KeyError: If a named task is not in the bundle.
    """
    if group in GROUPS:
        return list(GROUPS[group])
    chosen = [name.strip() for name in group.split(",") if name.strip()]
    unknown = [name for name in chosen if name not in BUNDLE]
    if unknown:
        raise KeyError(
            f"unknown task(s) {', '.join(unknown)}; have {', '.join(BUNDLE)} "
            f"or a group: {', '.join(GROUPS)}"
        )
    return chosen


def get(task: str) -> BundleTask:
    """
    :param task: Ladder name.

    :returns: Its bundle entry.

    :raises KeyError: If the task is not in the bundle.
    """
    try:
        return BUNDLE[task]
    except KeyError:
        raise KeyError(f"no eval ladder for {task!r}; have {', '.join(BUNDLE)}") from None


def resolve(task: str, rungs: str = "all", *, root: Optional[str] = None) -> List[Tuple[str, Path]]:
    """
    Turn ``--task`` and ``--rungs`` into files to grade.

    :param task: Ladder name.
    :param rungs: ``"all"`` or a comma list of labels.
    :param root: Bundle root; see :func:`bundle_root`.

    :returns: ``(rung label, absolute path)``, ascending by context length.

    :raises KeyError: If the task, or one of the requested rungs, does not exist. Requesting a rung
        a task does not have is an error rather than a silent skip: a missing row in a results table
        is read as "the model scored nothing there", not as "that cell was never run".
    """
    entry = get(task)
    base = bundle_root(root)
    available = dict(entry.rungs)

    if rungs == "all":
        wanted = entry.labels
    else:
        wanted = [r.strip() for r in rungs.split(",") if r.strip()]
        missing = [r for r in wanted if r not in available]
        if missing:
            raise KeyError(
                f"{task} has no rung(s) {', '.join(missing)}; it has {', '.join(entry.labels)}"
            )
    return [(label, base / available[label]) for label in wanted]


def describe(tasks: Sequence[str] = ()) -> List[str]:
    """
    :param tasks: Ladder names; empty means every one.

    :returns: Human-readable lines for ``--list-tasks``.
    """
    lines = []
    for name in tasks or list(BUNDLE):
        entry = get(name)
        lines.append(f"  {entry.name:<16} [{entry.group}]  spec={entry.spec}")
        lines.append(f"  {'':<16} rungs: {', '.join(entry.labels)}")
        if entry.note:
            lines.append(f"  {'':<16} note:  {entry.note}")
    return lines
