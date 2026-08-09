"""
The task registry: the single place that says what a task *is*.

Every task in the suite is described by exactly one :class:`TaskSpec`, and both halves of the
pipeline read it — the generator that writes an example, and the grader that scores a model's
answer. That shared reading is the point. Historically the prompt format, the answer parser, and the
gold-index convention were re-stated independently in the data generator, the native evaluator, and
the vLLM evaluator, and every one of the following bugs was one copy disagreeing with another:

* ``parse_doc_ids`` required both brackets in one copy and not the other, so a checkpoint that
  emitted the primed continuation ``8]`` scored 0.16 while one that emitted ``[15]`` scored 0.96 --
  on the same underlying capability.
* The grouping-JSON parser was fixed in one copy while the other kept falling through to a
  digit-scrape that lumped every doc id into one cluster (0.44 vs 0.82 on chunked grouping @2k).
* Gold indices are 1-based for contradiction and 0-based for outlier/rerank/nq, which lived only in
  people's heads and produced an off-by-one that read as a modelling result.

A task registered here cannot disagree with itself, because there is only one of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

__all__ = ["TaskSpec", "register", "get", "names", "clear"]


@dataclass(frozen=True)
class TaskSpec:
    """
    Everything the pipeline needs to know about one task.

    :param name: Registry key, e.g. ``"contradiction"``. Matches the ``--task`` CLI value.
    :param gold_index_base: ``0`` or ``1`` -- whether this task's ``gold_doc_indices`` count the
        first document as 0 or as 1. Contradiction is 1-based; outlier, rerank and nq are 0-based.
        Getting this wrong shifts every gold id by one and looks exactly like a weak model.
    :param build_prompt: ``(example, **opts) -> str``. Renders one example into a model prompt.
    :param parse: ``(text, n_docs) -> parsed answer | None``. Extracts the answer from a generation.
        Returns ``None`` when nothing parseable was produced, which callers must distinguish from a
        parsed-but-wrong answer -- collapsing the two hides decoding regressions as accuracy drops.
    :param score: ``(parsed, gold) -> dict[str, float]``. The metric(s) for this task.
    :param answer_is_set: True when the answer is an unordered set of document ids (scored with
        set-F1) rather than a ranked list or free text.
    :param description: One line, shown by ``ctc-eval --list-tasks``.
    """

    name: str
    gold_index_base: int
    build_prompt: Callable[..., str]
    parse: Callable[..., object]
    score: Callable[..., Dict[str, float]]
    answer_is_set: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.gold_index_base not in (0, 1):
            raise ValueError(
                f"task {self.name!r}: gold_index_base must be 0 or 1, got {self.gold_index_base!r}"
            )


_REGISTRY: Dict[str, TaskSpec] = {}


def register(spec: TaskSpec) -> TaskSpec:
    """
    Add a task to the registry.

    :param spec: The task description.

    :returns: The same spec, so this can be used as a module-level statement.

    :raises ValueError: If a task with this name is already registered. Silent replacement is
        never what you want here -- it is how two definitions of one task start coexisting.
    """
    if spec.name in _REGISTRY:
        raise ValueError(f"task {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> TaskSpec:
    """
    Look up a registered task.

    :param name: The task name.

    :returns: Its :class:`TaskSpec`.

    :raises KeyError: If the task is not registered, listing what is.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(names()) or "(none registered)"
        raise KeyError(f"unknown task {name!r}; registered tasks: {known}") from None


def names() -> List[str]:
    """:returns: Sorted names of every registered task."""
    return sorted(_REGISTRY)


def clear(name: Optional[str] = None) -> None:
    """
    Drop one task, or all of them. For tests only.

    :param name: Task to drop; ``None`` drops everything.
    """
    if name is None:
        _REGISTRY.clear()
    else:
        _REGISTRY.pop(name, None)
