"""
``grouping_labeled`` from OpenAlex -- ``BUILD_MATRIX.md`` row 14.

Registration only: the construction is in :mod:`ctc.tasks.grouping_labeled.generate` and the corpus
in :mod:`ctc.data.sources.openalex`. Row 13 (plain ``grouping``) came out of the *same* generation
with a different convert-time task key and was dropped from the roster on 2026-08-03, so there is
no second generator to write -- only the labeled spec is registered.

.. note::
   **The compact pool must be pointed at explicitly.** There is no default path, because the two
   staged files are the whole corpus and a wrong one is not detectable from the output::

       ctc-data build --task grouping_labeled \\
           -C path=/scratch/users/prasann/ctc_suite_data_shared/openalex_compact.jsonl \\
           -C eval_path=/path/to/openalex_eval2024_fetch.jsonl --out /data/ctc/v3

   Omitting ``eval_path`` is a *degraded* build, not a broken one: the eval side then comes from
   the 2024+ tail of the training fetch, which is ~900 papers and too thin per concept value to
   build a 32k rung at coarse levels. The build reports it as skipped draws rather than as an
   error, so check the report.
"""

from __future__ import annotations

from ....data.generators.base import Generator
from ....data.sources import openalex as source
from ..generate import SOURCE_TAG, build_example

__all__ = ["GENERATOR"]

GENERATOR = Generator(
    name="grouping_labeled",
    task="grouping_labeled",
    source=SOURCE_TAG,
    build_example=build_example,
    defaults={"num_docs": 22, "levels": "0,1,2,3", "min_per_group": 1},
    corpus=source.load_pool,
    corpus_defaults={
        "path": None,
        "eval_path": None,
        "eval_year_min": source.EVAL_YEAR_MIN,
        "max_papers": 0,
    },
    # The level must be a function of the example counter, not of the RNG: see the trap-14 warning
    # in ctc.tasks.grouping_labeled.generate.
    indexed=True,
    # Gold is a partition of EVERY document, so the shrink has no distractor to drop and a shorter
    # rung is a different partition.
    shrink_safe=False,
    notes="OpenAlex abstracts partitioned by concept level, each group named (names not scored)",
)
