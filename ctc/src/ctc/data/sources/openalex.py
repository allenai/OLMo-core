"""
OpenAlex scientific abstracts, indexed by concept level -- the substrate for ``grouping_labeled``.

A paper carries a best concept at each of OpenAlex's four hierarchy levels, L0 (about 19 top-level
fields) through L3 (thousands of narrow concepts). Choosing which level supplies the gold partition
is what sets a grouping example's granularity, so the pool's real product is not "papers" but a
**per-level index**: concept value -> the papers tagged with it, filtered to papers that have a
value at *every* level so any level can be used as the axis with no missing data.

.. note::
   **The corpus is a compact projection, and it is already on disk.** OpenAlex's bulk works
   snapshot is ~300 GB and is *not* staged; what is staged is the compact JSONL the pre-migration
   ``--api-fetch``/``--preprocess`` passes produced -- one line per usable paper, abstract already
   reconstructed from OpenAlex's inverted index and truncated to 120 words. Two files:

   * ``/scratch/users/prasann/ctc_suite_data_shared/openalex_compact.jsonl`` -- 52,000 papers,
     2018-2025, the pool ``BUILD_MATRIX.md`` row 13's "BLOCKER B2" was cleared with;
   * ``.../debug/ctc_fix_regen/grouping/openalex_eval2024_fetch.jsonl`` in the pre-migration tree
     -- 31,200 papers, 2024-2026, a *year-restricted* second fetch.

   This module reads that projection and does not re-implement the fetch: it is a network pass over
   the OpenAlex API measured in hours, and rerunning it would produce a different pool, not the one
   every shipped grouping file was built from.

.. warning::
   **Train and eval are split by publication YEAR, not by paper.** A 2019 paper and a 2023 paper on
   the same narrow concept are near-duplicates of each other; a random split puts both sides of
   such a pair across the boundary and the eval partition is one the model has effectively seen.
   The temporal split is the pre-migration leak control (``--eval-year-min 2024``) and it is why a
   *second* pool exists: the held-out slice of a single 2018-2025 fetch is only ~900 papers, far
   too thin per concept value to support a 32k rung at coarse levels, where there are only 19
   distinct values in the entire taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Paper", "OpenAlexPool", "load_pool", "LEVELS"]

#: The concept levels a paper must have a value at. Requiring all four is what lets one pool serve
#: every granularity: a paper missing L3 would silently drop out of the fine-grained levels only,
#: making the coarse and fine indices cover different corpora.
LEVELS: Tuple[int, ...] = (0, 1, 2, 3)

#: Year from which papers are held out for eval. The pre-migration ``--eval-year-min`` default.
EVAL_YEAR_MIN = 2024


@dataclass(frozen=True)
class Paper:
    """
    One abstract and its place in the concept hierarchy.

    :param id: OpenAlex work id. Used for de-duplication; never emitted.
    :param title: Paper title. Rendered by the grouping serializer, so it is part of the context.
    :param abstract: Abstract body, already truncated where the compact projection truncated it.
    :param year: Publication year -- the train/eval axis.
    :param concepts: level -> best (highest-scoring) concept value at that level.
    """

    id: str
    title: str
    abstract: str
    year: int
    concepts: Dict[int, str]


@dataclass
class OpenAlexPool:
    """
    Papers plus the per-level concept indices built over them.

    Not frozen, and not by oversight: :meth:`level_index` memoises. The index is a full pass over
    tens of thousands of papers and every example needs one, so rebuilding it per draw would
    dominate a build. One pool object is resolved per build and reused across every example, so the
    cache is per-split and cannot leak between them.

    :param papers: The training-side pool, in file order.
    :param eval_papers: A separate pool for the held-out years, or ``None`` to split ``papers`` by
        year alone. See the module warning on why a second pool exists.
    :param eval_year_min: Papers from this year onward belong to eval.
    :param provenance: What produced the pool; copied into every example's ``_meta``.
    """

    papers: Tuple[Paper, ...]
    eval_papers: Optional[Tuple[Paper, ...]] = None
    eval_year_min: int = EVAL_YEAR_MIN
    provenance: Dict[str, object] = field(default_factory=dict)
    _indices: Dict[Tuple[int, int], Dict[str, Tuple[Paper, ...]]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __len__(self) -> int:
        return len(self.papers)

    def for_split(self, split: str) -> "OpenAlexPool":
        """
        Slice by publication year.

        :param split: ``"train"`` or ``"eval"``.

        :returns: A pool holding only that split's papers, with a fresh index cache.

        :raises ValueError: On an unknown split name, or when the requested side is empty -- which
            means the year cut and the fetched pool disagree, and silently building from the wrong
            side is exactly the contamination this split exists to prevent.
        """
        if split == "train":
            kept = tuple(p for p in self.papers if p.year < self.eval_year_min)
        elif split == "eval":
            source = self.eval_papers if self.eval_papers is not None else self.papers
            kept = tuple(p for p in source if p.year >= self.eval_year_min)
        else:
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        if not kept:
            raise ValueError(
                f"the {split!r} side of this OpenAlex pool is empty at "
                f"eval_year_min={self.eval_year_min}; pass an eval_path whose papers are from "
                "those years, or move the cut"
            )
        return OpenAlexPool(
            papers=kept,
            eval_papers=self.eval_papers,
            eval_year_min=self.eval_year_min,
            provenance=self.provenance,
        )

    def level_index(self, level: int, min_per_value: int = 1) -> Dict[str, Tuple[Paper, ...]]:
        """
        Group this pool's papers by their concept value at one level.

        :param level: Concept level, 0 (coarsest) to 3 (finest).
        :param min_per_value: Drop values backed by fewer papers than this. A value that cannot
            fill even the smallest group is not a usable category, and leaving it in makes the
            capacity arithmetic in the generator think it has more room than it does.

        :returns: concept value -> its papers. Memoised per ``(level, min_per_value)``.
        """
        key = (level, min_per_value)
        cached = self._indices.get(key)
        if cached is not None:
            return cached
        buckets: Dict[str, List[Paper]] = {}
        for paper in self.papers:
            # Only papers annotated at EVERY level, so switching the axis never switches corpus.
            if len(paper.concepts) < len(LEVELS):
                continue
            value = paper.concepts.get(level)
            if value:
                buckets.setdefault(value, []).append(paper)
        index = {v: tuple(ps) for v, ps in buckets.items() if len(ps) >= min_per_value}
        self._indices[key] = index
        return index


def _paper(row: Dict) -> Optional[Paper]:
    """
    :param row: One line of a compact OpenAlex JSONL.

    :returns: The paper, or ``None`` when it is unusable -- no title, no abstract or no year. The
        compact projection already applied these filters; re-applying them here means a
        hand-assembled or partially-written pool cannot inject a document with empty text, which
        :func:`ctc.data.schema.make_document` would reject much later.
    """
    title = (row.get("title") or "").strip()
    abstract = (row.get("abstract") or "").strip()
    year = row.get("year")
    if not title or not abstract or not isinstance(year, int):
        return None
    concepts = {}
    for level in LEVELS:
        value = row.get(f"concept_L{level}")
        if value:
            concepts[level] = value
    return Paper(id=str(row.get("id") or title), title=title, abstract=abstract, year=year, concepts=concepts)


def load_pool(
    *,
    path: Optional[str] = None,
    eval_path: Optional[str] = None,
    eval_year_min: int = EVAL_YEAR_MIN,
    max_papers: int = 0,
) -> OpenAlexPool:
    """
    Read the compact OpenAlex projection.

    :param path: Compact JSONL. **Required at call time** -- there is no default, because the
        staged files *are* the corpus and a build against the wrong one is not detectable from its
        output.
    :param eval_path: A second compact JSONL supplying the held-out years. Strongly recommended;
        see the module warning.
    :param eval_year_min: Papers from this year onward go to eval.
    :param max_papers: Cap on papers read from ``path``, for a quick smoke build. ``0`` reads all.

    :returns: The pool.

    :raises ValueError: If ``path`` is unset, or no usable paper was read.
    """
    from ..io import load_jsonl

    if not path:
        raise ValueError(
            "grouping_labeled needs a compact OpenAlex pool: pass `-C path=<compact.jsonl>` "
            "(and, for a sound temporal split, `-C eval_path=<year-restricted compact.jsonl>`). "
            "The full OpenAlex works snapshot is ~300 GB and is deliberately not fetched here; "
            "see the module docstring for the two staged files."
        )
    papers = tuple(p for p in (_paper(row) for row in load_jsonl(path)) if p is not None)
    if max_papers:
        papers = papers[:max_papers]
    evals: Optional[Tuple[Paper, ...]] = None
    if eval_path:
        evals = tuple(p for p in (_paper(row) for row in load_jsonl(eval_path)) if p is not None)
    if not papers:
        raise ValueError(f"no usable papers in {path}")
    return OpenAlexPool(
        papers=papers,
        eval_papers=evals,
        eval_year_min=eval_year_min,
        provenance={
            "corpus": path,
            "eval_corpus": eval_path,
            "papers": len(papers),
            "eval_papers": len(evals) if evals is not None else None,
            "eval_year_min": eval_year_min,
        },
    )


def deep_papers(papers: Sequence[Paper]) -> List[Paper]:
    """
    :param papers: Any papers.

    :returns: Those annotated at every level in :data:`LEVELS` -- the only ones a grouping example
        may use. Exposed for auditing a pool's real capacity, which is what
        ``BUILD_MATRIX.md``'s "cap" printout reported.
    """
    return [p for p in papers if len(p.concepts) >= len(LEVELS)]
