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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_ROOT",
    "DEFAULT_BUNDLE",
    "Bundle",
    "BUNDLES",
    "BundleTask",
    "BUNDLE",
    "GROUPS",
    "XLONG_RUNGS",
    "bundle_root",
    "get",
    "get_bundle",
    "names",
    "resolve",
]

_WEKA = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"

#: Default bundle name. ``_clean`` rather than the original ``_eval_bundle_eval500_v2``: the
#: contradiction rungs in that one were calibrated against a filler pool that turned out to be
#: 92-99% FEVER/wiki rather than PubMed, so every contradiction rung overshot its label by ~1.8x.
#: The clean bundle has been the default since 2026-07-29.
DEFAULT_BUNDLE = "v2_clean"

DEFAULT_ROOT = f"{_WEKA}/_eval_bundle_eval500_v2_clean"

#: Environment variable overriding the default, so a local run points at a staged copy without
#: editing anything. Accepts a bundle NAME or a path.
ROOT_ENV = "CTC_EVAL_BUNDLE"

#: Ultra-long rung labels, kept apart from the 2k-32k ladder because they are **opt-in**: a 256k
#: rung is hours per task, and asking for ``--rungs all`` should not silently start one.
XLONG_RUNGS: Tuple[str, ...] = ("64k", "128k", "256k", "512k", "1M", "2M")


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
    :param eval_size: Rows per rung, **counted on weka** rather than inferred from the filename --
        several filenames disagree with their contents (``..._eval_600.jsonl`` holds 600, but
        ``..._fever_plain_n100_k3.jsonl`` holds 599 and ``nq_validation_k20_600.jsonl`` holds 500).
        Recorded so a sub-500 ladder can be flagged before a run rather than discovered in the
        results, and ``0`` means uncounted.
    :param note: Anything a reader of the numbers has to know.
    """

    name: str
    spec: str
    rungs: Tuple[Tuple[str, str], ...]
    group: str = "main"
    eval_size: int = 0
    note: str = ""

    @property
    def labels(self) -> List[str]:
        """:returns: This task's rung labels, ascending."""
        return [label for label, _ in self.rungs]

    @property
    def small_eval_warning(self) -> str:
        """
        :returns: A warning when this ladder is below the 500-example floor, else ``""``. A small
            eval inflates noise into apparent findings, so the caution travels with the task.
        """
        if not self.eval_size or self.eval_size >= 500:
            return ""
        se = (0.7 * 0.3 / self.eval_size) ** 0.5
        return (
            f"eval_size={self.eval_size}, below the 500 floor: quote the size and its error bar "
            f"(about ±{se:.3f} at f1 0.7) inline next to every number"
        )


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
        eval_size=500,
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
        eval_size=500,
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
        eval_size=600,
    ),
    "rerank": BundleTask(
        name="rerank",
        spec="rerank",
        rungs=(
            ("3k", "rerank/msmarco_trainhn_eval_k20_500.jsonl"),
            ("8k", "rerank/msmarco_trainhn_eval_k50_500.jsonl"),
            ("16k", "rerank/msmarco_trainhn_eval_k100_500.jsonl"),
        ),
        eval_size=500,
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
        eval_size=500,
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
        eval_size=648,
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
        eval_size=299,
        note="Held-out BEIR retrieval; graded by the same retrieval spec as nq.",
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
        eval_size=600,
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
        eval_size=599,
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


#: Ultra-long rungs, per bundle. **These are not the same files across bundles**, and the difference
#: is not cosmetic: ``_v2`` puts contradiction's 64k rung at n=1602 while ``_v2_clean`` puts it at
#: n=1525, because the clean rebuild recalibrated against a PubMed-only filler pool after the
#: original was found to be 92-99% FEVER/wiki. Quoting a "64k" number without saying which bundle it
#: came from compares two different eval sets.
#:
#: Recorded per bundle rather than derived, because there is no formula -- each is a measured
#: calibration. Rows land here as they are listed on weka; an absent rung raises rather than guesses.
_XLONG_V2: Dict[str, Dict[str, str]] = {
    "contradiction": {
        "64k": "contra/contradiction_eval_pubmed_both_n1602_k3_xlong_64k.jsonl",
        "128k": "contra/contradiction_eval_pubmed_both_n3204_k3_xlong_128k.jsonl",
        "256k": "contra/contradiction_eval_pubmed_both_n6408_k3_xlong_256k.jsonl",
        "512k": "contra/contradiction_eval_pubmed_both_n9888_k3_xlong_512k.jsonl",
        "1M": "contra/contradiction_eval_pubmed_both_n19775_k3_xlong_1M.jsonl",
        "2M": "contra/contradiction_eval_pubmed_both_n39550_k3_xlong_2M.jsonl",
    },
    "nq": {
        "64k": "nq/nq_validation_k418_xlong_64k.jsonl",
        "128k": "nq/nq_validation_k835_xlong_128k.jsonl",
        "256k": "nq/nq_validation_k1670_xlong_256k.jsonl",
        "512k": "nq/nq_validation_k3342_xlong_512k.jsonl",
        "1M": "nq/nq_validation_k6683_xlong_1M.jsonl",
        "2M": "nq/nq_validation_k13366_xlong_2M.jsonl",
    },
    "rerank": {
        "64k": "rerank/msmarco_trainhn_eval_k777_xlong_64k.jsonl",
        "128k": "rerank/msmarco_trainhn_eval_k1553_xlong_128k.jsonl",
        "256k": "rerank/msmarco_trainhn_eval_k3106_xlong_256k.jsonl",
        "512k": "rerank/msmarco_trainhn_eval_k6121_xlong_512k.jsonl",
        "1M": "rerank/msmarco_trainhn_eval_k12242_xlong_1M.jsonl",
        "2M": "rerank/msmarco_trainhn_eval_k24485_xlong_2M.jsonl",
    },
    "outlier": {
        "64k": "outlier/outlier_wiki100w_n448_k3_eval_xlong_64k.jsonl",
        "128k": "outlier/outlier_wiki100w_n897_k3_eval_xlong_128k.jsonl",
        "256k": "outlier/outlier_wiki100w_n1793_k3_eval_xlong_256k.jsonl",
        "512k": "outlier/outlier_wiki100w_n3605_k3_eval_xlong_512k.jsonl",
        "1M": "outlier/outlier_wiki100w_n7209_k3_eval_xlong_1M.jsonl",
        "2M": "outlier/outlier_wiki100w_n14419_k3_eval_xlong_2M.jsonl",
    },
}

#: The clean bundle's own recalibrated xlong rungs. Only the rows verified on weka are listed;
#: rerank's 1M/2M were carried over unchanged from ``_v2``, which is why those two share a filename.
_XLONG_V2_CLEAN: Dict[str, Dict[str, str]] = {
    "contradiction": {
        "64k": "contra/contradiction_eval_pubmed_both_n1525_k3_xlong_64k.jsonl",
        "128k": "contra/contradiction_eval_pubmed_both_n3051_k3_xlong_128k.jsonl",
        "256k": "contra/contradiction_eval_pubmed_both_n6102_k3_xlong_256k.jsonl",
        "512k": "contra/contradiction_eval_pubmed_both_n12204_k3_xlong_512k.jsonl",
        "1M": "contra/contradiction_eval_pubmed_both_n24408_k3_xlong_1M.jsonl",
        "2M": "contra/contradiction_eval_pubmed_both_n48816_k3_xlong_2M.jsonl",
    },
    "rerank": {
        "128k": "rerank/msmarco_trainhn_eval_k1530_xlong_128k.jsonl",
        "256k": "rerank/msmarco_trainhn_eval_k3061_xlong_256k.jsonl",
        "1M": "rerank/msmarco_trainhn_eval_k12242_xlong_1M.jsonl",
        "2M": "rerank/msmarco_trainhn_eval_k24485_xlong_2M.jsonl",
    },
}


#: The fast bundle's own ladders, written by ``src/scripts/ctc/build_fast_bundle.py``. Filenames
#: encode the construction, not the corpus: ``_tail10`` is prefix+tail with a 10% per-query tail
#: (90% of the prefill shared), ``_mux`` is query-multiplexed (100% of the documents shared, though
#: with ``query_position="both"`` almost none of that is a *token* prefix).
#:
#: Only three tasks are here. ``outlier`` needs per-document topic labels its eval files strip, and
#: the clustering that recovers them passes its exactness gate 2/25 of the time at 32k and 0/25
#: below -- so it cannot be built at these lengths without regenerating with labels. ``oolong``
#: builds from a source split that has no ultra-long member.
_FAST_LADDERS: Dict[str, Dict[str, str]] = {
    "contradiction": {
        "8k": "contradiction/rung_8192_tail10.jsonl",
        "16k": "contradiction/rung_16384_tail10.jsonl",
        "32k": "contradiction/rung_32768_tail10.jsonl",
        "64k": "contradiction/rung_65536_tail10.jsonl",
        "128k": "contradiction/rung_131072_tail10.jsonl",
        "256k": "contradiction/rung_262144_tail10.jsonl",
        "512k": "contradiction/rung_524288_tail10.jsonl",
        "1M": "contradiction/rung_1048576_tail10.jsonl",
    },
    "nq": {
        "8k": "nq/rung_8192_mux.jsonl",
        "16k": "nq/rung_16384_mux.jsonl",
        "32k": "nq/rung_32768_mux.jsonl",
        "64k": "nq/rung_65536_mux.jsonl",
        "128k": "nq/rung_131072_mux.jsonl",
        "256k": "nq/rung_262144_mux.jsonl",
        "512k": "nq/rung_524288_mux.jsonl",
        "1M": "nq/rung_1048576_mux.jsonl",
    },
    # Generated from the wiki100w article pool, not transformed from a reliable rung: the planted
    # construction has to know each document's topic, and the eval files strip that. The rungs
    # match the reliable ones in corpus size but share no documents with them.
    #
    # ⚠ The 8k rung leaks. The answer's topic is the one candidate the tail does not top up, and a
    # short corpus has too few topics for that absence to hide in: guessing among the topics missing
    # from the tail scores 0.203 there, against ~0.09 chance. It falls off fast with length --
    # 0.090 at 16k, 0.048 at 32k, 0.024 at 64k, 0.006 at 256k, 0.0015 at 1M -- so **use 32k and
    # above**, and never quote an 8k number from this bundle without the control beside it.
    "outlier": {
        "8k": "outlier/rung_8192_planted.jsonl",
        "16k": "outlier/rung_16384_planted.jsonl",
        "32k": "outlier/rung_32768_planted.jsonl",
        "64k": "outlier/rung_65536_planted.jsonl",
        "128k": "outlier/rung_131072_planted.jsonl",
        "256k": "outlier/rung_262144_planted.jsonl",
        "512k": "outlier/rung_524288_planted.jsonl",
        "1M": "outlier/rung_1048576_planted.jsonl",
    },
    # 100 rows at every rung -- a fifth of the 500 floor. Quote the size and its error bar
    # (about +/-0.046 at 0.7) inline next to any oolong number from this bundle.
    "oolong": {
        "8k": "oolong/rung_8192_mux.jsonl",
        "16k": "oolong/rung_16384_mux.jsonl",
        "32k": "oolong/rung_32768_mux.jsonl",
        "64k": "oolong/rung_65536_mux.jsonl",
        "128k": "oolong/rung_131072_mux.jsonl",
        "256k": "oolong/rung_262144_mux.jsonl",
        "512k": "oolong/rung_524288_mux.jsonl",
        "1M": "oolong/rung_1048576_mux.jsonl",
    },
    # Note the 32k rung, which the RELIABLE rerank ladder does not have: its CE-filtered pool caps
    # at 100 documents per query, but pooling makes the corpus size a free parameter, so a fast
    # rerank rung reaches 32k like every other task.
    "rerank": {
        "8k": "rerank/rung_8192_mux.jsonl",
        "16k": "rerank/rung_16384_mux.jsonl",
        "32k": "rerank/rung_32768_mux.jsonl",
        "64k": "rerank/rung_65536_mux.jsonl",
        "128k": "rerank/rung_131072_mux.jsonl",
        "256k": "rerank/rung_262144_mux.jsonl",
        "512k": "rerank/rung_524288_mux.jsonl",
        "1M": "rerank/rung_1048576_mux.jsonl",
    },
}


@dataclass(frozen=True)
class Bundle:
    """
    A named collection of eval files on disk.

    More than one exists because eval data gets rebuilt, and a rebuilt rung is a *different eval
    set* rather than a corrected copy of the same one. Naming them makes "which data produced this
    number" answerable from the results file instead of from memory.

    :param name: What ``--bundle`` takes.
    :param root: Directory the task paths resolve against.
    :param kind: ``"reliable"`` -- one independently sampled corpus per row, so N rows are N
        independent measurements -- or ``"fast"``, where many queries share a corpus so a prefill
        can be reused. The two are **not interchangeable**: rebuilding an eval set to share corpora
        measurably moved scores (+0.215/+0.261 on outlier, -0.10..-0.18 on contradiction), with a
        control isolating the cause to document placement rather than to the sharing.
    :param xlong: Extra ultra-long rungs, per task.
    :param ladders: Tasks whose entire rung table this bundle supplies itself, replacing the base
        one. The fast bundle needs this: its files are named for the construction that built them,
        and it carries only the tasks and rungs that construction supports, so inheriting the base
        ladder would resolve to filenames that do not exist.
    :param ladder_version: The results-hub ``eval_version`` facet this bundle's numbers belong to,
        or ``None`` when it is not known. Only the registered bundles below can answer this: an
        ad-hoc ``--bundle /some/path`` may hold anything, so it leaves the field unset and the
        person ingesting has to say. Deriving it from :attr:`kind` instead would tag every staged
        or experimental bundle ``v2`` -- filing it into the reliable series, which is the precise
        mislabelling the ``_meta`` block exists to prevent.
    :param description: One line, shown by ``--list-bundles``.
    """

    name: str
    root: str
    kind: str = "reliable"
    xlong: Dict[str, Dict[str, str]] = field(default_factory=dict)
    ladders: Dict[str, Dict[str, str]] = field(default_factory=dict)
    ladder_version: Optional[str] = None
    description: str = ""

    def declares_own_ladder(self, task: str) -> bool:
        """:returns: Whether this bundle supplies ``task``'s rung table itself."""
        return task in self.ladders

    def rungs_for(self, task: str) -> Dict[str, str]:
        """
        :param task: Ladder name.

        :returns: ``{rung label: path relative to the root}`` -- this bundle's own table if it has
            one for the task, otherwise the base ladder plus this bundle's xlong rungs.
        """
        if task in self.ladders:
            return dict(self.ladders[task])
        merged = dict(BUNDLE[task].rungs)
        merged.update(self.xlong.get(task, {}))
        return merged


#: Every bundle this repo knows how to read.
BUNDLES: Dict[str, Bundle] = {
    "v2_clean": Bundle(
        name="v2_clean",
        root=f"{_WEKA}/_eval_bundle_eval500_v2_clean",
        xlong=_XLONG_V2_CLEAN,
        ladder_version="v2",
        description=(
            "Default. The v2 ladder with contradiction rebuilt against a PubMed-only filler pool "
            "and its rungs recalibrated."
        ),
    ),
    "v2": Bundle(
        name="v2",
        root=f"{_WEKA}/_eval_bundle_eval500_v2",
        xlong=_XLONG_V2,
        ladder_version="v2",
        description=(
            "The original v2 ladder, kept because existing results were produced against it -- "
            "including the 256k runs' xlong rungs. Its contradiction rungs overshoot their labels "
            "by ~1.8x (FEVER/wiki filler), so read contradiction numbers from it with that in mind."
        ),
    ),
    "fast": Bundle(
        name="fast",
        root=f"{_WEKA}/_eval_bundle_eval500_v2_fast",
        kind="fast",
        ladders=_FAST_LADDERS,
        ladder_version="fast",
        description=(
            "Shared-corpus rungs, 8k-1M. contradiction shares 90% of its prefill; nq and rerank "
            "share every document but almost no token prefix. NOT comparable to a reliable "
            "bundle -- the construction moves scores on its own."
        ),
    ),
}


def get_bundle(name_or_path: Optional[str] = None) -> Bundle:
    """
    Resolve ``--bundle``: a registered name, a filesystem path, or the default.

    A path is accepted so a staged local copy works without registering anything; it is treated as
    a ``reliable`` bundle carrying the base ladder only.

    :param name_or_path: A key of :data:`BUNDLES`, a directory path, or ``None``.

    :returns: The bundle.
    """
    raw = name_or_path or os.environ.get(ROOT_ENV) or DEFAULT_BUNDLE
    if raw in BUNDLES:
        return BUNDLES[raw]
    return Bundle(name=Path(raw).name or raw, root=raw, description="ad-hoc path")


def bundle_root(explicit: Optional[str] = None) -> Path:
    """
    :param explicit: A bundle name or path from the command line, if given.

    :returns: The root directory. Not checked for existence here -- a launcher resolves this on the
        submitting host, where the weka mount does not exist.
    """
    return Path(get_bundle(explicit).root)


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
    bundle = get_bundle(root)
    base = Path(bundle.root)
    available = bundle.rungs_for(task)

    if bundle.kind == "fast" and not bundle.declares_own_ladder(task):
        raise KeyError(
            f"bundle {bundle.name!r} has no {task}: a shared-corpus construction exists for "
            f"{', '.join(sorted(bundle.ladders))} only. Grade {task} against a reliable bundle."
        )

    if rungs == "all":
        # Base ladder only. The ultra-long rungs are opt-in because one 256k rung is hours per task,
        # and `--rungs all` must not silently start one.
        #
        # A bundle that supplies its own ladder is filtered to what it actually has: the fast
        # bundle starts at 8k, and asking for its 2k rung is a missing file, not a missing score.
        wanted = entry.labels
        if bundle.declares_own_ladder(task):
            wanted = [r for r in wanted if r in available]
    elif rungs == "xlong":
        wanted = [r for r in XLONG_RUNGS if r in available]
        if not wanted:
            raise KeyError(
                f"{task} has no ultra-long rungs in bundle {bundle.name!r}; "
                f"it has {', '.join(available)}"
            )
    else:
        wanted = [r.strip() for r in rungs.split(",") if r.strip()]
        missing = [r for r in wanted if r not in available]
        if missing:
            raise KeyError(
                f"{task} has no rung(s) {', '.join(missing)} in bundle {bundle.name!r}; "
                f"it has {', '.join(available)}"
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
        size = f"  eval_size={entry.eval_size}" if entry.eval_size else ""
        lines.append(f"  {entry.name:<16} [{entry.group}]  spec={entry.spec}{size}")
        lines.append(f"  {'':<16} rungs: {', '.join(entry.labels)}")
        if entry.note:
            lines.append(f"  {'':<16} note:  {entry.note}")
        if entry.small_eval_warning:
            lines.append(f"  {'':<16} ⚠      {entry.small_eval_warning}")
    return lines
