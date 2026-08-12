"""
Turning a generator into a train set and an eval ladder, with the contamination controls built in.

Three decisions live here rather than in each generator, because the pre-migration tree let every
generator own its own ``main()`` and they drifted -- five train/eval splitters with different
defaults, ``--eval-frac`` 0.1 in one file and 0.2 in the next.

**1. Train and eval draw from independent RNG substreams.** The old drivers threaded one
``random.Random(seed)`` through train and then eval, so changing ``--num-train`` silently changed
the eval set. An eval set that moves when you resize training is not a fixed measuring stick, and
the drift is invisible -- both files look fine. Streams are now keyed by ``(seed, split, rung)``,
so the eval set at a given seed is the same set forever.

**2. The eval ladder is nested, not re-generated per rung.** One canonical set is built at the
longest rung and shorter rungs are derived by *dropping distractors in place*. Every rung then
grades the same underlying questions with identical gold text, and each rung's documents are a
subset of the next one's. Regenerating per rung instead confounds "the context got longer" with
"the questions changed", which is the whole axis of these experiments.

**3. Contamination is checked at build time, not audited afterwards.** See :func:`fingerprint`.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from ..format.registry import TaskSpec
from . import ladders
from .generators import base as generators
from .schema import gold_field_for

__all__ = ["BuildReport", "fingerprint", "gold_fingerprint", "shrink", "build_eval", "build_train"]

#: Give up on a config that cannot produce fresh examples. Hitting this means the parameter space is
#: too small for the requested count -- a real problem, since the alternative is a training set of
#: near-duplicates that looks the right size.
MAX_REJECTS_PER_EXAMPLE = 50


@dataclass
class BuildReport:
    """
    What a build produced and what it threw away.

    :param task: Task name.
    :param split: ``"train"`` or ``"eval"``.
    :param counts: rung label -> examples written.
    :param duplicates: Examples discarded for repeating one already in this split.
    :param contaminated: Train examples discarded for reusing an eval example's gold. Non-zero is
        not an error -- it is the guard working -- but a large fraction means the gold space is too
        small for the requested sizes, and the remaining train set is more correlated with eval
        than the count suggests.
    :param skipped: Draws the generator declined -- a degenerate item set, a query with no
        answer-bearing passage, a pool too thin for the requested size. Reported because it is the
        difference between "the corpus is fine" and "the corpus ran out", which the row count alone
        cannot tell you.
    :param reused_pool: Times a corpus-backed generator wrapped back to the start of its pool.
        Non-zero means the same question appears in several examples with different distractors,
        so the split has fewer distinct questions than rows.
    :param notes: Anything a reader of the resulting files must know.
    """

    task: str
    split: str
    counts: Dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    contaminated: int = 0
    skipped: int = 0
    reused_pool: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        """:returns: Examples written across all rungs."""
        return sum(self.counts.values())

    def summary(self) -> str:
        """:returns: One line for a build log, plus any notes."""
        parts = [
            f"{self.duplicates} dup" if self.duplicates else "",
            f"{self.contaminated} contaminated" if self.contaminated else "",
            f"{self.skipped} skipped" if self.skipped else "",
            f"{self.reused_pool} pool wraps" if self.reused_pool else "",
        ]
        rejected = ", ".join(p for p in parts if p)
        line = f"{self.task}/{self.split}: {self.total} examples"
        if rejected:
            line += f" ({rejected})"
        return "\n".join([line, *(f"  ! {note}" for note in self.notes)])


def _digest(*parts: str) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def fingerprint(example: Mapping[str, Any]) -> str:
    """
    Identify an example by its content, independent of document order.

    Order-independent on purpose: the same corpus reshuffled is the same example for the purpose of
    "have I already generated this", and a shuffle-sensitive hash would let near-identical examples
    through while reporting zero duplicates.

    :param example: A unified-format example.

    :returns: A hex digest.
    """
    return _digest(*sorted(d["text"] for d in example["documents"]), *example.get("queries", []))


def gold_fingerprint(example: Mapping[str, Any], spec: TaskSpec) -> str:
    """
    Identify an example by *the answer*: the gold documents' text plus the query.

    This is the contamination-relevant identity. Two examples with different filler but the same
    gold structure -- the same cycle over the same names, the same claim pair -- are the same
    question, and one in train plus one in eval is leakage even though the full corpora differ and
    :func:`fingerprint` sees nothing.

    :param example: A unified-format example.
    :param spec: The task spec, for the gold field and index base.

    :returns: A hex digest.
    """
    docs = example["documents"]
    gold = example.get(gold_field_for(spec)) or []
    positions = sorted(_flatten(gold))
    texts = sorted(docs[p - spec.gold_index_base]["text"] for p in positions)
    return _digest(*texts, *example.get("queries", []))


def _flatten(gold: Sequence) -> List[int]:
    flat: List[int] = []
    for entry in gold:
        flat.extend(entry) if isinstance(entry, (list, tuple)) else flat.append(entry)
    return flat


def shrink(example: Mapping[str, Any], n_docs: int, spec: TaskSpec, rng: random.Random) -> Dict:
    """
    Derive a shorter version of an example by dropping distractors, keeping every gold document.

    Relative order is preserved, so the shorter rung's documents are a subsequence of the longer
    one's and the ladder is genuinely nested. Any ``_meta`` list parallel to ``documents``
    (textgroups' per-passage ``counts``, mathmatch's ``answer_values``) is shrunk alongside --
    leaving them full-length would make the metadata describe a corpus that no longer exists, and
    the CoT builders read exactly those arrays.

    :param example: The canonical, longest-rung example.
    :param n_docs: Target document count.
    :param spec: The task spec.
    :param rng: Seeded RNG for choosing which distractors to drop.

    :returns: The shortened example.

    :raises ValueError: If ``n_docs`` is larger than the example, or too small to hold its gold.
    """
    docs = example["documents"]
    field_name = gold_field_for(spec)
    gold = example.get(field_name) or []
    keep = {p - spec.gold_index_base for p in _flatten(gold)}
    if n_docs > len(docs):
        raise ValueError(f"cannot grow an example from {len(docs)} to {n_docs} documents")
    if n_docs < len(keep):
        raise ValueError(f"n_docs={n_docs} cannot hold {len(keep)} gold documents")

    droppable = [i for i in range(len(docs)) if i not in keep]
    dropped = set(rng.sample(droppable, len(docs) - n_docs))
    kept = [i for i in range(len(docs)) if i not in dropped]
    old_to_new = {old: new + spec.gold_index_base for new, old in enumerate(kept)}

    out = dict(example)
    out["documents"] = [docs[i] for i in kept]
    if gold:
        # Preserve the gold structure's SHAPE, not just its values. Wrapping a flat multi-gold list
        # into singleton groups is not cosmetic: `ctc.tasks._retrieval.flatten_gold` reads a nested
        # gold as one group per query and returns only the first, so a shrunk `hotpotqa` or `fiqa`
        # rung would be graded on one of its gold documents and marked wrong for the rest -- while
        # the longest rung, which never passes through here, was graded on all of them. The rungs
        # would then disagree for a reason that has nothing to do with context length.
        if any(isinstance(g, (list, tuple)) for g in gold):
            out[field_name] = sorted(
                sorted(old_to_new[p - spec.gold_index_base] for p in _flatten([g])) for g in gold
            )
        else:
            out[field_name] = sorted(old_to_new[p - spec.gold_index_base] for p in gold)
    if "hard_neg_indices" in out:
        out["hard_neg_indices"] = sorted(
            old_to_new[p - spec.gold_index_base]
            for p in out["hard_neg_indices"]
            if p - spec.gold_index_base in old_to_new
        )
    if isinstance(out.get("ce_scores"), list) and len(out["ce_scores"]) == len(docs):
        out["ce_scores"] = [out["ce_scores"][i] for i in kept]
    meta = out.get("_meta")
    if isinstance(meta, dict):
        out["_meta"] = {
            k: ([v[i] for i in kept] if isinstance(v, list) and len(v) == len(docs) else v)
            for k, v in meta.items()
        }
    return out


class _Cursor:
    """
    A split's running example counter, shared across its rungs.

    Corpus-backed generators take one question (or claim pair) per example, identified by this
    index rather than sampled from the RNG -- sampling would reuse some questions while leaving
    others untouched, inside a split that is supposed to cover its pool. Sharing one cursor across
    the rungs of a *train* build is what keeps the 20k examples 20k distinct questions rather than
    the same 4k asked at five lengths.

    :param size: Pool size, or ``None`` when the generator needs no index.
    :param report: Where a wrap-around is recorded.
    """

    def __init__(self, size: Optional[int], report: BuildReport) -> None:
        self.value = 0
        self.size = size
        self.report = report

    def take(self) -> int:
        """:returns: The next index, wrapping at the pool size and counting the wrap."""
        index = self.value
        self.value += 1
        if self.size:
            if index and index % self.size == 0:
                self.report.reused_pool += 1
            return index % self.size
        return index


def _cursor_for(
    generator: generators.Generator, resolved: Mapping[str, Any], report: BuildReport
) -> _Cursor:
    """
    :param generator: The generator.
    :param resolved: Its resolved config, possibly carrying a corpus.
    :param report: Where wrap-arounds are recorded.

    :returns: A cursor that wraps at the pool size only for generators that **consume** the pool
        one item per example. ``outlier``, ``outlier_review`` and ``oolong`` sample from theirs
        instead -- an article or a log item can back many examples -- so wrapping their counter
        would mean rebuilding the same rows every ``len(pool)`` examples.
    """
    if not generator.indexed:
        return _Cursor(None, report)
    corpus = resolved.get("corpus")
    try:
        size = len(corpus) or None
    except TypeError:
        size = None
    return _Cursor(size, report)


def _draw(
    generator: generators.Generator,
    config: Mapping[str, Any],
    rng: random.Random,
    *,
    count: int,
    seen: Set[str],
    forbidden_gold: Set[str],
    spec: TaskSpec,
    report: BuildReport,
    cursor: Optional[_Cursor] = None,
) -> Iterator[Dict]:
    """Yield ``count`` fresh examples, rejecting duplicates, eval-gold reuse and dead draws."""
    produced = 0
    rejects = 0
    while produced < count:
        kwargs = dict(config)
        if generator.indexed:
            kwargs["index"] = (cursor or _Cursor(None, report)).take()
        example = generator.build_example(rng, **kwargs)
        if example is None:
            report.skipped += 1
            rejects += 1
        elif fingerprint(example) in seen:
            report.duplicates += 1
            rejects += 1
        elif gold_fingerprint(example, spec) in forbidden_gold:
            report.contaminated += 1
            rejects += 1
        else:
            seen.add(fingerprint(example))
            produced, rejects = produced + 1, 0
            yield example
            continue
        if rejects >= MAX_REJECTS_PER_EXAMPLE:
            raise RuntimeError(
                f"{generator.name}: {MAX_REJECTS_PER_EXAMPLE} consecutive rejections at "
                f"{_describe(config)}. The parameter space or the corpus pool is too small for "
                f"{count} distinct examples; widen it rather than accepting near-duplicate data."
            )


def _describe(config: Mapping[str, Any]) -> str:
    """:param config: A resolved build config. :returns: It, minus the (huge) corpus pool."""
    return str({k: v for k, v in config.items() if k != "corpus"})


def _resolve(
    generator: generators.Generator,
    config: Optional[Mapping[str, Any]],
    corpus: Any,
    split: str,
) -> Dict[str, Any]:
    """
    Merge the per-example config with this split's slice of the corpus.

    :param generator: The generator.
    :param config: Caller overrides.
    :param corpus: A loaded pool, or ``None``.
    :param split: ``"train"`` or ``"eval"``.

    :returns: The resolved kwargs for ``build_example``.

    :raises ValueError: If the generator needs a corpus and none was supplied.
    """
    resolved = generator.config(**dict(config or {}))
    if generator.corpus is None:
        return resolved
    if corpus is None:
        raise ValueError(
            f"{generator.name} needs a corpus; call generator.load_corpus() and pass it in, or "
            "let ctc.data.cli do it"
        )
    # Train/eval separation is a property of the POOL, not of the loop over it: a pool that knows
    # how to split itself cannot be split two different ways by two callers.
    resolved["corpus"] = corpus.for_split(split) if hasattr(corpus, "for_split") else corpus
    return resolved


def build_eval(
    task: str,
    spec: TaskSpec,
    *,
    size: int = 500,
    seed: int = 7,
    rungs: Optional[Sequence[str]] = None,
    config: Optional[Mapping[str, Any]] = None,
    corpus: Any = None,
) -> Tuple[Dict[str, List[Dict]], BuildReport]:
    """
    Build an eval ladder, nested wherever the task allows it.

    Three constructions, and which one a task gets is declared on its generator, never guessed:

    * **shrink** (the default) -- one canonical set at the longest rung, shorter rungs derived by
      dropping distractors in place;
    * **a task's own ladder** (:attr:`~ctc.data.generators.base.Generator.build_ladder`) -- for
      ``outlier``, where dropping random distractors can destroy the "outlier is the rarest topic"
      invariant, so the majority is grown instead;
    * **independent rungs** -- for ``oolong``, whose gold is recomputed over the drawn items and so
      cannot survive any resize. The report says so, because the resulting ladder's rung-to-rung
      deltas carry eval-set resampling noise the others do not.

    :param task: Ladder name.
    :param spec: Its grading spec.
    :param size: Examples per rung. 500 is the suite floor; anything smaller must be flagged
        inline wherever its numbers are quoted, with an error bar.
    :param seed: Eval seed. Kept distinct from the train seed by default so the two never share a
        stream even if someone passes the same number.
    :param rungs: Rungs to emit; defaults to the task's full ladder.
    :param config: Generator parameter overrides.
    :param corpus: A loaded pool for corpus-backed generators. Defaults to loading one, which is
        the only step that touches the network.

    :returns: ``({rung label: examples}, report)``. The report carries the rejection counts,
        which are the guard's output -- a build that discards a third of its draws to contamination
        is not the same build as one that discards none, even though the files look identical.

    :raises ValueError: If ``size`` is below 500.
    """
    generator = generators.get(task)
    if size < 500:
        raise ValueError(
            f"eval_size={size} is below the suite floor of 500. A smaller eval inflates noise into "
            "apparent findings; if you need it, build it explicitly and flag the size and its "
            "error bar next to every number."
        )
    labels = list(rungs) if rungs else ladders.rungs_for(task)
    longest = labels[-1]
    report = BuildReport(task=task, split="eval")
    if corpus is None:
        corpus = generator.load_corpus()
    resolved = _resolve(generator, config, corpus, "eval")
    cursor = _cursor_for(generator, resolved, report)

    if generator.build_ladder is not None:
        out = _draw_ladder(generator, resolved, labels, task, size, seed, spec, report, cursor)
    elif not generator.shrink_safe:
        out = _draw_independent(generator, resolved, labels, task, size, seed, spec, report, cursor)
        report.notes.append(
            f"{task} rungs are generated independently: its gold is recomputed per draw, so no "
            "two rungs grade the same question and rung-to-rung deltas include eval-set noise"
        )
    else:
        resolved[generator.scaling_param] = ladders.docs_for_rung(task, longest)
        rng = random.Random(f"{seed}:eval:{longest}")
        canonical = list(
            _draw(
                generator,
                resolved,
                rng,
                count=size,
                seen=set(),
                forbidden_gold=set(),
                spec=spec,
                report=report,
                cursor=cursor,
            )
        )
        out = {longest: canonical}
        for label in labels[:-1]:
            n_docs = ladders.docs_for_rung(task, label)
            # One stream per rung, keyed by the rung, so adding a rung never perturbs the others.
            shrink_rng = random.Random(f"{seed}:eval:shrink:{label}")
            out[label] = [shrink(ex, n_docs, spec, shrink_rng) for ex in canonical]

    report.counts = {label: len(out[label]) for label in labels}
    return {label: out[label] for label in labels}, report


def _draw_independent(
    generator: generators.Generator,
    resolved: Dict[str, Any],
    labels: Sequence[str],
    task: str,
    size: int,
    seed: int,
    spec: TaskSpec,
    report: BuildReport,
    cursor: _Cursor,
) -> Dict[str, List[Dict]]:
    """Generate each rung on its own stream, for a generator that cannot resize an example."""
    out: Dict[str, List[Dict]] = {}
    for label in labels:
        rung_config = dict(resolved)
        rung_config[generator.scaling_param] = ladders.docs_for_rung(task, label)
        out[label] = list(
            _draw(
                generator,
                rung_config,
                random.Random(f"{seed}:eval:{label}"),
                count=size,
                seen=set(),
                forbidden_gold=set(),
                spec=spec,
                report=report,
                cursor=cursor,
            )
        )
    return out


def _draw_ladder(
    generator: generators.Generator,
    resolved: Dict[str, Any],
    labels: Sequence[str],
    task: str,
    size: int,
    seed: int,
    spec: TaskSpec,
    report: BuildReport,
    cursor: _Cursor,
) -> Dict[str, List[Dict]]:
    """Build every rung of a row at once, through the generator's own ladder builder."""
    sizes = {label: ladders.docs_for_rung(task, label) for label in labels}
    longest = max(sizes, key=lambda label: sizes[label])
    out: Dict[str, List[Dict]] = {label: [] for label in labels}
    seen: Set[str] = set()
    rejects = 0
    while len(out[longest]) < size:
        index = cursor.take()
        row = generator.build_ladder(
            random.Random(f"{seed}:eval:{index}"), index=index, rungs=sizes, **resolved
        )
        if row is None or fingerprint(row[longest]) in seen:
            if row is None:
                report.skipped += 1
            else:
                report.duplicates += 1
            rejects += 1
            if rejects >= MAX_REJECTS_PER_EXAMPLE:
                raise RuntimeError(
                    f"{generator.name}: {MAX_REJECTS_PER_EXAMPLE} consecutive rejections building "
                    f"the ladder at {_describe(resolved)}; the corpus pool is too small."
                )
            continue
        seen.add(fingerprint(row[longest]))
        rejects = 0
        for label in labels:
            out[label].append(row[label])
    return out


def build_train(
    task: str,
    spec: TaskSpec,
    *,
    total: int = 20_000,
    seed: int = 42,
    rungs: Optional[Sequence[str]] = None,
    config: Optional[Mapping[str, Any]] = None,
    eval_examples: Sequence[Mapping[str, Any]] = (),
    corpus: Any = None,
) -> Tuple[List[Dict], BuildReport]:
    """
    Build a training set spread uniformly over the rung ladder's document counts.

    Uniform over rungs rather than fixed at one size: a model trained only at the short end has
    never seen the long-context regime it is then measured in, and the ladder is the axis of the
    experiment.

    :param task: Ladder name.
    :param spec: Its grading spec.
    :param total: Total examples, split equally across rungs.
    :param seed: Train seed.
    :param rungs: Rungs to cover; defaults to the task's full ladder.
    :param config: Generator parameter overrides.
    :param eval_examples: The eval set, if built. Any train example reusing an eval example's gold
        is rejected -- pass this, or the guard cannot run.
    :param corpus: A loaded pool for corpus-backed generators; defaults to loading one.

    :returns: ``(examples, report)``, the examples in rung order.

    :raises ValueError: If the ladder is held out. Training on ``fiqa``, ``scifact``,
        ``outlier_review`` or ``contra_fever`` destroys the one property they exist for, so this is
        an error rather than a warning -- by the time a warning is noticed the checkpoint is
        trained and the OOD column is meaningless.
    """
    generator = generators.get(task)
    if generator.eval_only:
        raise ValueError(
            f"{task} is a held-out ladder: it exists to measure generalisation to an unseen "
            "corpus, and training on it makes every number from it in-distribution. Build train "
            f"data from the matching in-distribution ladder instead ({generator.task})."
        )
    labels = list(rungs) if rungs else ladders.rungs_for(task)
    per_rung = total // len(labels)
    forbidden = {gold_fingerprint(ex, spec) for ex in eval_examples}

    report = BuildReport(task=task, split="train")
    if corpus is None:
        corpus = generator.load_corpus()
    base = _resolve(generator, config, corpus, "train")
    # One cursor for the whole split, not one per rung: restarting it per rung would ask the same
    # questions five times at five lengths instead of covering the pool.
    cursor = _cursor_for(generator, base, report)

    seen: Set[str] = set()
    examples: List[Dict] = []
    for label in labels:
        resolved = dict(base)
        resolved[generator.scaling_param] = ladders.docs_for_rung(task, label)
        rng = random.Random(f"{seed}:train:{label}")
        drawn = list(
            _draw(
                generator,
                resolved,
                rng,
                count=per_rung,
                seen=seen,
                forbidden_gold=forbidden,
                spec=spec,
                report=report,
                cursor=cursor,
            )
        )
        report.counts[label] = len(drawn)
        examples.extend(drawn)
    return examples, report
