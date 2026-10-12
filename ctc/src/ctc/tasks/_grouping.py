"""
Shared machinery for the grouping tasks: grouping and grouping_labeled.

Partition a set of abstracts into K topical categories. Scored **pairwise** -- on which documents
were put together -- rather than by comparing cluster labels, because cluster identity is
arbitrary: a model that finds exactly the right partition but numbers the clusters differently is
completely correct, and a label-wise comparison would score it near zero.

``grouping_labeled`` additionally asks the model to name each group. The names are not scored; the
task exists to test whether being asked to articulate the shared topic changes the partition.

**Gold is 0-based** and converted to the 1-based ids the prompt renders.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Sequence

from ..format import assemble, metrics, parsing
from ..format.registry import TaskSpec

__all__ = ["parse", "score", "build_labeled_target", "make_grouping_spec"]


def parse(text: str, n_docs: int) -> Optional[List[List[int]]]:
    """
    :param text: Raw model generation.
    :param n_docs: Corpus size, bounding the digit-scrape fallback.

    :returns: Clusters of 1-indexed ids, or ``None``.
    """
    return parsing.parse_partition(text, n_docs)


#: Zero score for an unparseable partition. Named so the shape cannot drift from what :func:`score`
#: returns on the success path.
_ZERO = {
    "pairwise_precision": 0.0,
    "pairwise_recall": 0.0,
    "pairwise_f1": 0.0,
    "k_exact": 0.0,
    "coverage": 0.0,
    "parsed": 0.0,
}


def score(
    parsed: Optional[Sequence[Sequence[int]]],
    gold: Sequence[Sequence[int]],
    n_docs: Optional[int] = None,
) -> Dict[str, float]:
    """
    Score a predicted partition.

    :param parsed: Output of :func:`parse`. ``None`` returns zeros without touching ``gold``, so
        the call is safe before any example is in hand.
    :param gold: Gold clusters, 0-based.
    :param n_docs: Corpus size. Defaults to the number of documents implied by ``gold``, which is
        correct whenever the gold partition covers the corpus -- as it does for this task, where
        every document belongs to exactly one group.

    :returns: ``pairwise_precision``, ``pairwise_recall``, ``pairwise_f1``, ``k_exact``,
        ``coverage``, ``parsed``, plus ``ari``/``nmi`` when scikit-learn is installed.
    """
    if parsed is None:
        return dict(_ZERO)
    if n_docs is None:
        n_docs = sum(len(c) for c in gold)
    n = n_docs
    gold_clusters = [[int(i) + 1 for i in c] for c in gold]
    gold_labels = parsing.partition_to_labels(gold_clusters, n)
    pred_labels = parsing.partition_to_labels(list(parsed), n)
    out = dict(metrics.pairwise_metrics(pred_labels, gold_labels))

    # k_exact: did the model produce the requested number of groups? coverage: what fraction of
    # documents it actually placed. A model can score well pairwise while quietly dropping
    # documents, and coverage is what makes that visible.
    out["k_exact"] = float(len(parsed) == len(gold))
    placed = {d for c in parsed for d in c if 1 <= d <= n}
    out["coverage"] = len(placed) / n if n else 0.0
    out["parsed"] = 1.0
    out.update(metrics.clustering_extras(pred_labels, gold_labels))
    return out


def build_labeled_target(example: Dict) -> str:
    """
    Build the ``grouping_labeled`` target.

    :param example: A unified-format example. Group names come from ``cluster_labels`` -- note the
        field name; a missing or short list yields empty labels rather than raising, matching the
        pre-migration builder.

    :returns: The JSON grouping object with a ``label`` per group, ids 1-based.
    """
    labels = example.get("cluster_labels") or []
    groups = []
    for i, cluster in enumerate(example["gold_doc_indices"]):
        groups.append(
            {
                "label": labels[i] if i < len(labels) else "",
                "doc_ids": [int(d) + 1 for d in cluster],
            }
        )
    return json.dumps({"groups": groups})


def make_grouping_spec(
    *,
    name: str,
    instruction: str,
    description: str,
    rungs: tuple,
    query_builder: Callable[[Dict], str],
    sources: tuple = (),
) -> TaskSpec:
    """
    Build the spec for one grouping task.

    :param name: Task name.
    :param instruction: The instruction string, verbatim.
    :param description: One line for ``--list-tasks``.
    :param rungs: The task's ladder.
    :param query_builder: ``(example) -> str``; the positioned ask.
    :param sources: Source corpora.

    :returns: The spec, not yet registered.
    """

    def build_prompt(example: Dict, **opts) -> str:
        """
        :param example: A unified-format example.
        :param opts: Assembly options. ``query_position`` is accepted and **ignored**, matching the
            legacy path this reproduces.

        :returns: The prompt.
        """
        # Discarded, not forwarded. The spec declares honors_query_position=False, and the
        # fingerprint pins the field to "after" for such tasks so that two runs differing only in
        # an inert knob still compare as compatible. Letting the knob through made it NOT inert:
        # a --query-position before run produced a genuinely different prompt while writing a
        # fingerprint that claimed "after", i.e. the guard would certify a format the checkpoint
        # was never trained on. outlier and qdmatch already discarded it; this one did not.
        opts.pop("query_position", None)
        return assemble.assemble(
            example,
            task=name,
            unified=False,
            header=instruction,
            positioned=query_builder(example),
            query_position="after",
            **opts,
        )

    return TaskSpec(
        name=name,
        description=description,
        gold_index_base=0,
        instruction=instruction,
        serializer=name,
        unified=False,
        honors_query_position=False,  # legacy path hardcodes documents-then-query
        rungs=rungs,
        build_prompt=build_prompt,
        parse=parse,
        score=score,
        primary_metric="pairwise_f1",
        # A full partition of a large corpus is long, and EOS is genuinely emitted here, so any
        # text stop would cut a valid multi-line answer short.
        #
        # 4096, not the pre-migration 2048. This answer grows with BOTH the rung and the concept
        # level: at the 32k rung an L3 example asks for near-singleton clusters, so the target is
        # roughly one labelled group per document. Measured over a build at n=175, the target's
        # median is 1392 Qwen3 tokens but its **maximum is 2597** -- so at 2048 the finest examples
        # of the longest rung are cut off, `parse_partition` returns a partition missing its tail,
        # and coverage (with it pairwise F1) collapses for a reason that is not the model's.
        max_new_tokens=4096,
        stop="eos",
        answer_is_set=False,
        sources=sources,
    )
