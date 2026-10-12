"""
The generator registry: how :mod:`ctc.data` finds a task's data builder.

A generator declares two things -- where its raw material comes from, and how to construct **one
example** from a seeded RNG. Everything above that (how many, which rungs, train vs eval,
deduplication, auditing) is shared and lives in :mod:`ctc.data.build`, because those are exactly
the decisions that must not be re-litigated per task. The pre-migration tree let each generator own
its own ``main()``, and they diverged: five different train/eval splitters with different defaults,
``--eval-frac`` of 0.1 in one file and 0.2 in another, ``int(round(x))`` in one and bare ``int(x)``
in the next.

Discovery is lazy. ``ctc.tasks.<...>`` is imported only when someone asks to build that task, so
installing the package for evaluation does not drag in corpus loaders, BM25 or a cross-encoder --
``pip install ./ctc`` promises no GPU and no compiler, and the generation side is where that
promise is easiest to break. Corpus modules keep the same discipline internally: ``datasets``,
``pyserini`` and ``transformers`` are imported inside the loader that needs them, never at module
scope, so every generator module here imports on a bare install.

**A generator is keyed by its LADDER name, not its task name.** ``nq``, ``fiqa`` and ``scifact``
are three ladders graded by one ``retrieval`` spec; ``contra_fever`` is graded by the
``contradiction`` spec. Keying by spec would collapse them, and a results table that cannot tell
``contradiction`` from ``contra_fever`` cannot report an in-distribution-vs-held-out gap at all.
The names here are exactly :data:`ctc.eval.bundles.BUNDLE`'s, so ``ctc-data build --task nq`` and
``ctc-eval --task nq`` mean the same thing.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

__all__ = ["Generator", "GENERATORS", "get", "names", "corpus_free_names"]

#: Ladder name -> ``"module"`` or ``"module:ATTRIBUTE"`` holding its :class:`Generator`. Listed
#: rather than discovered, for the same reason :data:`ctc.tasks.TASK_MODULES` is: a task that
#: vanishes from a suite build because its module failed to import should be an error, not a
#: quietly shorter results table.
GENERATORS: Dict[str, str] = {
    # ── pure synthetic: no corpus, no network, byte-reproducible from a seed ──
    "cycle": "ctc.tasks.cycle.generate",
    "groups4": "ctc.tasks.groups4.generate",
    "mathmatch": "ctc.tasks.mathmatch.generate",
    "strmatch": "ctc.tasks.strmatch.generate",
    "textgroups": "ctc.tasks.textgroups.generate",
    # ── the five in-distribution CTC-suite ladders ──
    "contradiction": "ctc.tasks.contradiction.sources.pubmed",
    "redundancy": "ctc.tasks.redundancy.generate",
    "nq": "ctc.tasks.retrieval.sources.nq",
    "outlier": "ctc.tasks.outlier.sources.wiki100w",
    "rerank": "ctc.tasks.rerank.generate",
    "oolong": "ctc.tasks.oolong.generate",
    # ── further in-distribution corpora graded by an existing spec ──
    "hotpotqa": "ctc.tasks.retrieval.sources.hotpotqa",
    # ── the absence family: what is NOT here rather than what is ──
    "absence": "ctc.tasks.absence.sources.gutenberg",
    "xabsence": "ctc.tasks.xabsence.sources.pubmed",
    # ── ladders whose gold covers every document, so their rungs cannot nest ──
    "reorder": "ctc.tasks.reorder.sources.gutenberg",
    "grouping_labeled": "ctc.tasks.grouping_labeled.sources.openalex",
    # ── qdmatch: one spec, two corpora, two suite rows (BUILD_MATRIX.md 21a/21b) ──
    "qdmatch_nq": "ctc.tasks.qdmatch.generate:NQ",
    "qdmatch_hpqa": "ctc.tasks.qdmatch.generate:HOTPOTQA",
    # ── the four held-out (OOD) ladders: same contracts, different corpora ──
    "fiqa": "ctc.tasks.retrieval.sources.beir:FIQA",
    "scifact": "ctc.tasks.retrieval.sources.beir:SCIFACT",
    "outlier_review": "ctc.tasks.outlier.sources.review",
    "contra_fever": "ctc.tasks.contradiction.sources.fever",
}


@dataclass(frozen=True)
class Generator:
    """
    How to build one example of one ladder.

    :param name: Ladder name -- the key in :data:`GENERATORS` and in
        :data:`ctc.eval.bundles.BUNDLE`. Not always the task name.
    :param task: The registered :class:`~ctc.format.registry.TaskSpec` that grades it.
    :param source: The ``source`` tag every emitted example carries.
    :param build_example: ``(rng, **config) -> example | None``. Must consume the RNG
        deterministically and must not read global state -- the ladder builder calls it with
        derived substreams and relies on the same ``(seed, config)`` giving the same example.
        Returning ``None`` means "this draw was degenerate, skip it"; the builder advances and
        counts the skip rather than silently emitting a malformed row.
    :param defaults: Default construction parameters, overridden per build.
    :param scaling_param: The config key that sets corpus size, i.e. the one a rung ladder varies.
        ``"num_docs"`` for almost everything; ``oolong`` scales a *token* budget instead, because
        its items are lines inside one document.
    :param shrink_safe: True when dropping non-gold documents from an example cannot create or
        destroy a gold answer. This is what allows the eval ladder to be *nested* -- one canonical
        set of examples built at the largest rung, with smaller rungs derived by removing
        distractors, so every rung grades the same underlying questions. False forces either a
        :attr:`build_ladder` of the task's own or independent per-rung generation, which costs
        cross-rung comparability.
    :param corpus: ``(**corpus_config) -> pool``, called **once** per build. ``None`` for the
        synthetic tasks. The pool is passed to ``build_example`` as ``corpus=``; a test can inject
        a small fixture pool instead and never touch the network.
    :param corpus_defaults: Default corpus-loading parameters (cache paths, pool sizes, HF splits).
        Kept apart from :attr:`defaults` because they are not per-example knobs and must not be
        swept.
    :param indexed: ``build_example`` also takes ``index=``, a running example counter. Needed by
        every corpus-backed generator: one example consumes one question or claim pair, and picking
        it from the RNG instead would repeat questions inside a split. The index keeps each example
        a pure function of ``(rng, index, pool)``, so a build can still be sharded or resumed.
    :param build_ladder: ``(rng, *, rungs, **config) -> {label: example} | None``, for tasks whose
        rungs cannot be derived by dropping distractors. Only ``outlier`` needs it, and the reason
        is a correctness fix, not tidiness -- see :mod:`ctc.tasks.outlier.generate`.
    :param eval_only: This ladder is held out. Building training data from it would destroy the
        one property it exists for, so the builder refuses rather than warns.
    :param notes: Free text surfaced by ``ctc-data list``.
    """

    name: str
    task: str
    source: str
    build_example: Callable[..., Optional[Dict[str, Any]]]
    defaults: Dict[str, Any] = field(default_factory=dict)
    scaling_param: str = "num_docs"
    shrink_safe: bool = True
    corpus: Optional[Callable[..., Any]] = None
    corpus_defaults: Dict[str, Any] = field(default_factory=dict)
    indexed: bool = False
    build_ladder: Optional[Callable[..., Any]] = None
    eval_only: bool = False
    notes: str = ""

    @property
    def nested_ladder(self) -> bool:
        """
        :returns: Whether this generator's rungs grade the same questions over nested corpora.
            False means each rung is an independent draw, so the ladder's x-axis carries eval-set
            resampling noise on top of the length effect -- true of ``oolong``, whose gold is
            recomputed over whichever items were drawn.
        """
        return self.shrink_safe or self.build_ladder is not None

    def _merge(
        self, defaults: Mapping[str, Any], overrides: Mapping[str, Any], kind: str
    ) -> Dict[str, Any]:
        unknown = [k for k, v in overrides.items() if v is not None and k not in defaults]
        if unknown:
            raise TypeError(
                f"{self.name}: unknown {kind} parameter(s) {sorted(unknown)}; "
                f"accepts {sorted(defaults)}"
            )
        merged = dict(defaults)
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return merged

    def config(self, **overrides: Any) -> Dict[str, Any]:
        """
        Merge overrides onto the per-example defaults.

        :param overrides: Caller-supplied parameters. ``None`` values are ignored so a CLI can pass
            unset options through without clobbering a default.

        :returns: The resolved config.

        :raises TypeError: On a parameter this generator does not accept -- a typo in a build
            script would otherwise be silently ignored and produce data at the default size.
        """
        return self._merge(self.defaults, overrides, "build")

    def corpus_config(self, **overrides: Any) -> Dict[str, Any]:
        """
        Merge overrides onto the corpus-loading defaults.

        :param overrides: Caller-supplied parameters; ``None`` values are ignored.

        :returns: The resolved corpus config.

        :raises TypeError: On a parameter this generator's loader does not accept.
        """
        return self._merge(self.corpus_defaults, overrides, "corpus")

    def load_corpus(self, **overrides: Any) -> Any:
        """
        Load this generator's raw material. The only step that touches the network.

        :param overrides: Corpus-config overrides.

        :returns: The pool object, or ``None`` for a synthetic generator.
        """
        if self.corpus is None:
            return None
        return self.corpus(**self.corpus_config(**overrides))


def names() -> List[str]:
    """:returns: Every ladder with a ported generator."""
    return list(GENERATORS)


def corpus_free_names() -> List[str]:
    """
    :returns: Ladders whose generator needs no corpus -- the ones a test can build end to end with
        no network, no cache and no HF download.
    """
    return [name for name in GENERATORS if get(name).corpus is None]


def get(task: str) -> Generator:
    """
    Load one ladder's generator.

    :param task: Ladder name.

    :returns: Its :class:`Generator`.

    :raises KeyError: If the ladder has no ported generator.
    :raises AttributeError: If the module exists but exposes no such generator.
    :raises ValueError: If the generator's own ``name`` disagrees with its registry key, which
        means a module was copied and only half-edited.
    """
    if task not in GENERATORS:
        raise KeyError(
            f"no generator ported for {task!r}; have {', '.join(GENERATORS)}. "
            "See records/data-generator-port.md for what remains un-ported."
        )
    target = GENERATORS[task]
    module_path, _, attribute = target.partition(":")
    module = importlib.import_module(module_path)
    generator: Optional[Generator] = getattr(module, attribute or "GENERATOR", None)
    if generator is None:
        raise AttributeError(f"{module_path} defines no {attribute or 'GENERATOR'}")
    if generator.name != task:
        raise ValueError(
            f"{module_path} declares name={generator.name!r} but is registered as {task!r}"
        )
    return generator
