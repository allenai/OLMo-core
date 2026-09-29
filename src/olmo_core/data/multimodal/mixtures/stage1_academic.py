"""The academic QA source group for Molmo2 stage 1 (``Molmo2-Stage1.py --academic_rate``).

Stage 2's large synthetic QA sets, trained in stage 1 as well, so a stage 1 that is mixed into
text midtraining has more distinct images to see than PixMo-Cap's ~717k. Every source is a
training split, and the user turn is the bare question behind the source's style tag
(``plot_qa: What is the ratio of ...``), as in the rest of stage 1; see
:class:`~olmo_core.data.multimodal.academic_dataset.Stage1AcademicDatasetConfig`.

**Which sets.** The training set of a benchmark the stage-2 checkpoints are evaluated on stays
stage-2 data (:data:`~olmo_core.data.multimodal.academic.registry.STAGE2_EVAL_TRAIN_SETS`): VQAv2,
TextVQA, ChartQA, DocVQA, InfographicVQA, AI2D and A-OKVQA. So does TallyQA, whose images overlap
the VQAv2 eval's. What is left of stage 2's single-image academic data, by training rows (one per
image):

========================  =========  ===================================================
source                    rows       share of the group (default sources, real sizes)
========================  =========  ===================================================
CoSyn, 7 categories       357,202    77.1%, split by sqrt(size)
``plot_qa``               157,070    7.4%; weighted as 20,000 rows; 20 questions / image
``dv_qa``                 200,000    5.2%; weighted as 10,000 rows
``figure_qa``             100,000    5.2%; weighted as 10,000 rows
``pixmo_clocks``          800,269    5.0%: capped (:attr:`Stage1AcademicSource.max_share`)
========================  =========  ===================================================

**Two constraints shape the split**, on top of sqrt(size):

* **Row caps** (:attr:`Stage1AcademicSource.weighting_size_cap`), stage 2's ``image_only_v9``
  ``root_size_factor``: a source is weighted as if it had at most that many rows. DVQA, FigureQA
  and PlotQA are templated charts with a handful of question forms; uncapped, their row counts
  would give them 44% of the group rather than 18%.
* **Share caps** (:attr:`Stage1AcademicSource.max_share`): a source takes at most that fraction of
  the group's rate, and what it gives up goes to the uncapped sources in proportion to their
  weights (:func:`academic_group_fractions`). PixMo-Clocks is one narrow skill, reading a clock
  face, and even at its row cap its 800k synthetic rows would take 21.6% of the group; it is held
  to 5%, no more than one templated chart set gets. Its images are then never repeated in a run.

OKVQA, ST-VQA, ScienceQA and TabMWP are small (6k-25k rows) and are left out; ScienceQA and AI2D
are multiple choice, which the stage-1 prompt family has no port of.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.data.multimodal.academic.registry import STAGE2_EVAL_TRAIN_SETS
from olmo_core.data.multimodal.academic_dataset import (
    Stage1AcademicDataset,
    Stage1AcademicDatasetConfig,
)
from olmo_core.exceptions import OLMoConfigurationError

__all__ = [
    "Stage1AcademicSource",
    "STAGE1_ACADEMIC_SOURCES",
    "STAGE1_ACADEMIC_SOURCE_NAMES",
    "DEFAULT_ACADEMIC_SOURCES",
    "STAGE2_EVAL_TRAIN_SETS",
    "academic_weighting_sizes",
    "check_share_caps",
    "academic_group_fractions",
    "build_stage1_academic_source",
]


@dataclass(frozen=True)
class Stage1AcademicSource:
    """How one academic source is sampled in the stage-1 group."""

    weighting_size_cap: Optional[int] = None
    """Row count the group weights the source by at most (stage 2's ``root_size_factor``)."""
    max_share: Optional[float] = None
    """Largest fraction of the group's rate the source may take; see
    :func:`academic_group_fractions`."""
    max_questions: Optional[int] = None
    """Questions trained per image per epoch (:attr:`Stage1AcademicDatasetConfig.max_questions`)."""


#: 20 questions of a PlotQA image fit a 2,560-token sequence: over 300 sampled images, p99 2,360
#: tokens and max 2,437 with the Qwen3 tokenizer and 8 crops (at 24, 1.3% overflow). The images
#: average 131 questions, so an image's questions are spread over its epochs.
PLOT_QA_MAX_QUESTIONS = 20

STAGE1_ACADEMIC_SOURCES: Dict[str, Stage1AcademicSource] = {
    "cosyn_chart_exp": Stage1AcademicSource(),
    "cosyn_chemical_exp": Stage1AcademicSource(),
    "cosyn_diagram_exp": Stage1AcademicSource(),
    "cosyn_document": Stage1AcademicSource(),
    "cosyn_math_exp": Stage1AcademicSource(),
    "cosyn_music_exp": Stage1AcademicSource(),
    "cosyn_table_exp": Stage1AcademicSource(),
    "dv_qa": Stage1AcademicSource(weighting_size_cap=10_000),
    "figure_qa": Stage1AcademicSource(weighting_size_cap=10_000),
    "plot_qa": Stage1AcademicSource(weighting_size_cap=20_000, max_questions=PLOT_QA_MAX_QUESTIONS),
    "pixmo_clocks": Stage1AcademicSource(weighting_size_cap=250_000, max_share=0.05),
}

STAGE1_ACADEMIC_SOURCE_NAMES: Tuple[str, ...] = tuple(STAGE1_ACADEMIC_SOURCES)

DEFAULT_ACADEMIC_SOURCES: Tuple[str, ...] = STAGE1_ACADEMIC_SOURCE_NAMES


def _source(name: str) -> Stage1AcademicSource:
    if name in STAGE2_EVAL_TRAIN_SETS:
        raise OLMoConfigurationError(
            f"{name!r} is the training set of a stage-2 eval benchmark "
            f"({STAGE2_EVAL_TRAIN_SETS[name]}), so it is not trained in stage 1"
        )
    if name not in STAGE1_ACADEMIC_SOURCES:
        raise OLMoConfigurationError(
            f"Unknown stage-1 academic source {name!r}; expected one of "
            f"{STAGE1_ACADEMIC_SOURCE_NAMES}"
        )
    return STAGE1_ACADEMIC_SOURCES[name]


def academic_weighting_sizes(names: Sequence[str], sizes: Sequence[int]) -> List[int]:
    """The sizes the group's sqrt(size) split weights its sources by: each row count, capped at
    the source's :attr:`Stage1AcademicSource.weighting_size_cap`.

    :param names: Stage-1 academic source names.
    :param sizes: Their row counts, parallel to ``names``.

    :raises OLMoConfigurationError: If a name is not a stage-1 academic source.
    """
    out = []
    for name, size in zip(names, sizes):
        cap = _source(name).weighting_size_cap
        out.append(int(size) if cap is None else min(int(size), cap))
    return out


def check_share_caps(names: Sequence[str]) -> None:
    """Check that the sources in ``names`` can share the whole group's rate under their
    :attr:`Stage1AcademicSource.max_share` caps.

    :raises OLMoConfigurationError: If every source is capped and the caps sum to less than 1 (e.g.
        ``pixmo_clocks`` alone), since the rate would then have nowhere to go; or a name is not a
        stage-1 academic source.
    """
    caps = [_source(n).max_share for n in names]
    if names and all(c is not None for c in caps) and sum(caps) < 1.0:  # type: ignore[arg-type]
        raise OLMoConfigurationError(
            f"academic sources {list(names)} are all share-capped ({caps}), and the caps sum to "
            "less than the whole group: add a source without a max_share"
        )


def academic_group_fractions(names: Sequence[str], weights: Sequence[float]) -> List[float]:
    """Split the group's rate among ``names``, starting from ``weights`` (sqrt of the capped
    sizes) and holding each source to its :attr:`Stage1AcademicSource.max_share`.

    A source whose proportional share exceeds its cap is set to the cap; the rest of the rate is
    re-divided among the other sources in proportion to their weights, and the check repeats
    until no uncapped source is over its cap.

    :param names: Stage-1 academic source names.
    :param weights: Their unnormalized weights, parallel to ``names``, all > 0.

    :returns: One fraction per name, summing to 1.

    :raises OLMoConfigurationError: See :func:`check_share_caps`.
    """
    check_share_caps(names)
    w = np.asarray(weights, dtype=np.float64)
    share_caps = [_source(n).max_share for n in names]
    caps = np.array([1.0 if c is None else c for c in share_caps], dtype=np.float64)
    fixed = np.zeros(len(names), dtype=bool)
    frac = w / w.sum()
    for _ in range(len(names)):
        over = ~fixed & (frac > caps + 1e-12)
        if not over.any():
            break
        fixed |= over
        frac[fixed] = caps[fixed]
        free = ~fixed
        if free.any():
            frac[free] = w[free] / w[free].sum() * (1.0 - caps[fixed].sum())
    return [float(f) for f in frac]


def build_stage1_academic_source(
    name: str, tokenizer, *, max_crops: int = 8, seed: int = 0
) -> Stage1AcademicDataset:
    """Build one stage-1 academic source by name.

    :raises OLMoConfigurationError: If ``name`` is a stage-2 eval training set or unknown.
    """
    src = _source(name)
    return Stage1AcademicDatasetConfig(
        name=name, max_crops=max_crops, max_questions=src.max_questions, seed=seed
    ).build(tokenizer)
