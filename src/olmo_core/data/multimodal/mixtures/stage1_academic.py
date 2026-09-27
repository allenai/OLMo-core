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

========================  =========  ==========================================================
source                    rows       in stage 1
========================  =========  ==========================================================
CoSyn, 7 categories       357,202    default; split by sqrt(size) like the rest of the group
``dv_qa``                 200,000    default; weighted as 10,000 rows (stage 2's cap)
``plot_qa``               157,070    default; weighted as 20,000 rows; 20 questions per image
``figure_qa``             100,000    default; weighted as 10,000 rows
``pixmo_clocks``          800,269    registered, not default; weighted as 250,000 rows
========================  =========  ==========================================================

The weighting caps are stage 2's (``image_only_v9`` ``root_size_factor``): DVQA, FigureQA and
PlotQA are templated charts with a handful of question forms, and uncapped their row counts would
give them 44% of the group rather than 19%. PixMo-Clocks is one narrow skill (reading a clock
face) whose 800k synthetic rows would take 22% of the group at its stage-2 cap and 33% without
it; it is selectable with ``--academic_sources`` for an ablation. OKVQA, ST-VQA, ScienceQA and TabMWP are small
(6k-25k rows) and are left out; ScienceQA and AI2D are multiple choice, which the stage-1 prompt
family has no port of.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    "NON_DEFAULT_ACADEMIC_SOURCES",
    "DEFAULT_ACADEMIC_SOURCES",
    "STAGE2_EVAL_TRAIN_SETS",
    "academic_weighting_sizes",
    "build_stage1_academic_source",
]


@dataclass(frozen=True)
class Stage1AcademicSource:
    """How one academic source is sampled in the stage-1 group."""

    weighting_size_cap: Optional[int] = None
    """Row count the group weights the source by at most (stage 2's ``root_size_factor``)."""
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
    "pixmo_clocks": Stage1AcademicSource(weighting_size_cap=250_000),
}

STAGE1_ACADEMIC_SOURCE_NAMES: Tuple[str, ...] = tuple(STAGE1_ACADEMIC_SOURCES)

#: Registered for ablations, but not in a default run (see the module doc).
NON_DEFAULT_ACADEMIC_SOURCES: Tuple[str, ...] = ("pixmo_clocks",)

DEFAULT_ACADEMIC_SOURCES: Tuple[str, ...] = tuple(
    n for n in STAGE1_ACADEMIC_SOURCE_NAMES if n not in NON_DEFAULT_ACADEMIC_SOURCES
)


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
