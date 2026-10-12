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

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .fingerprint import FormatFingerprint, hash_prompt

__all__ = ["TaskSpec", "register", "get", "names", "clear"]


@dataclass(frozen=True)
class TaskSpec:
    """
    Everything the pipeline needs to know about one task.

    Declared once, in ``ctc/tasks/<name>/spec.py``, and read by the generator that writes examples,
    the grader that scores answers, and the fingerprint that checks the two agree. That last one is
    why some fields here restate things the code already encodes: recording the *declared* format
    lets :meth:`fingerprint` be derived rather than hand-maintained, so a spec and its fingerprint
    cannot drift apart.

    :param name: Registry key, e.g. ``"contradiction"``. Matches the ``--task`` CLI value.
    :param gold_index_base: ``0`` or ``1`` -- whether this task's ``gold_doc_indices`` count the
        first document as 0 or as 1. Contradiction is 1-based; outlier, rerank and nq are 0-based.
        Getting this wrong shifts every gold id by one and looks exactly like a weak model.
    :param build_prompt: ``(example, **opts) -> str``. Renders one example into a model prompt.
    :param parse: ``(text, n_docs) -> parsed answer | None``. Extracts the answer from a generation.
        Returns ``None`` when nothing parseable was produced, which callers must distinguish from a
        parsed-but-wrong answer -- collapsing the two hides decoding regressions as accuracy drops.
    :param score: ``(parsed, gold) -> dict[str, float]``. The metric(s) for this task.
    :param instruction: The task's instruction string, verbatim. Hashed into the fingerprint, so
        editing it correctly invalidates every checkpoint trained under the old wording.
    :param instruction_variants: Additional instruction strings this task can use, when the choice
        depends on the *example* rather than the task. retrieval picks among three (single-doc,
        multi-doc, multi-query) by inspecting query and gold counts; qa and cot_retrieval pick
        among two. All of them are hashed into the fingerprint, because a checkpoint is bound to
        every wording it was trained under, not just the most common one -- hashing only
        ``instruction`` would let a variant be edited without invalidating anything.
    :param honors_query_position: Whether ``query_position`` actually changes this task's prompt.
        False for ``grouping``, ``grouping_labeled`` and ``outlier``, which take a legacy path that
        hardcodes documents-then-query and never consults the knob. Recorded because an **inert**
        knob must not produce a false fingerprint mismatch: two runs differing only in a setting
        that does nothing are compatible, and flagging them would teach people to ignore the guard.
    :param serializer: Key into the table in :mod:`ctc.format.documents`, or ``"default"``.
    :param unified: Whether this task uses the unified prompt shape -- a generic alpaca header with
        the task instruction positioned alongside the documents. True for tasks with **no
        per-example query**, where the instruction ("find every contradicting pair") *is* the whole
        ask: 11 of the 20 canonical tasks. False for tasks that carry a real question per example
        (retrieval, qa, oolong, grouping, outlier, rerank, summarization, cot_retrieval,
        grouping_labeled), which keep the task instruction in the header and position the question.

        In the pre-migration code this was a hardcoded ``force_unified`` set inside ``build_prompt``,
        and a separate ``unified_prompt`` argument that could force it on for any task -- but that
        argument was never once enabled, so the hardcoded set was the whole behaviour. Declaring it
        per task makes the split visible and lets the fingerprint record it.
    :param rungs: This task's own context-length ladder. There is no global set -- nq runs 2k-8k
        while the xlong ladders reach 512k -- so ``--rungs all`` is only answerable per task.
    :param primary_metric: Which key of ``score``'s output is *the* number for this task. Named
        explicitly so a results table cannot quietly switch between f1 and exact_match.
    :param max_new_tokens: Default decode budget. Too small truncates a correct answer into a parse
        failure, which reads as a capability limit rather than as a config mistake.

        The pre-migration tables disagreed here: ``eval_lc_native_docchunk``'s ``TASK_CFG`` gave
        contradiction ``max_new=200`` while ``run_rung_eval``'s ``TASK_MAX_NEW`` gave 512, with the
        driver passing its value explicitly -- so the budget in force depended on which entry point
        you used. Declaring it once is the point of this field.
    :param stop: Key into :data:`ctc.eval.stopping.STOP_PRESETS`, governing when generation ends and
        what survives for the parser. A surprising share of this project's bad numbers came from
        getting that wrong rather than from the model.
    :param answer_is_set: True when the answer is an unordered set of ids (scored with set-F1)
        rather than a ranked list or free text.
    :param sources: Named source corpora this task can be built from, e.g.
        ``("pubmed", "fever", "wiki")``. Empty for purely synthetic tasks.
    :param description: One line, shown by ``ctc-eval --list-tasks``.
    :param extra: Task-specific knobs that no other task shares. Kept out of the fields above so
        the common contract stays small.
    """

    name: str
    gold_index_base: int
    build_prompt: Callable[..., str]
    parse: Callable[..., object]
    score: Callable[..., Dict[str, float]]
    instruction: str = ""
    instruction_variants: Tuple[str, ...] = ()
    serializer: str = "default"
    unified: bool = False
    honors_query_position: bool = True
    rungs: Tuple[str, ...] = ()
    primary_metric: str = "f1"
    max_new_tokens: int = 512
    stop: str = "eos"
    answer_is_set: bool = False
    sources: Tuple[str, ...] = ()
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gold_index_base not in (0, 1):
            raise ValueError(
                f"task {self.name!r}: gold_index_base must be 0 or 1, got {self.gold_index_base!r}"
            )
        if self.max_new_tokens <= 0:
            raise ValueError(f"task {self.name!r}: max_new_tokens must be positive")

    def fingerprint(self, **overrides: Any) -> FormatFingerprint:
        """
        Derive this task's format fingerprint.

        Everything defining the format is already declared on the spec, so the fingerprint is
        computed from it rather than written out a second time -- which is what stops the two from
        disagreeing. The caller supplies only what the spec cannot know: the build options
        (``query_position``), the tokenizer, the marker ids, the chunk layout, and the document-id
        range actually present in the built data.

        :param overrides: Fingerprint fields to set. ``query_position`` should always be passed --
            it is a real build option that varies across runs, and its default here is only the
            most common value, not a safe assumption.

        :returns: The fingerprint to write beside the shards or the checkpoint.
        """
        from .documents import ITEM_SEPARATOR

        base: Dict[str, Any] = dict(
            task=self.name,
            prompt_hash=hash_prompt(self.instruction, *self.instruction_variants, self.serializer),
            serializer=self.serializer,
            item_separator=ITEM_SEPARATOR,
            gold_index_base=self.gold_index_base,
            prompt_shape="unified" if self.unified else "classic",
        )
        if not self.honors_query_position:
            # The knob does nothing for this task, so pin it: otherwise two compatible runs that
            # merely passed different values would compare as incompatible.
            base["query_position"] = "after"
        base.update(overrides)
        return FormatFingerprint(**base)


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
