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
    """

    task: str
    split: str
    counts: Dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    contaminated: int = 0

    @property
    def total(self) -> int:
        """:returns: Examples written across all rungs."""
        return sum(self.counts.values())

    def summary(self) -> str:
        """:returns: One line for a build log."""
        rejected = (
            f", {self.duplicates} dup, {self.contaminated} contaminated"
            if (self.duplicates or self.contaminated)
            else ""
        )
        return f"{self.task}/{self.split}: {self.total} examples{rejected}"


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
        out[field_name] = sorted(
            sorted(old_to_new[p - spec.gold_index_base] for p in _flatten([g])) for g in gold
        )
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
) -> Iterator[Dict]:
    """Yield ``count`` fresh examples, rejecting duplicates and eval-gold reuse."""
    produced = 0
    rejects = 0
    while produced < count:
        example = generator.build_example(rng, **config)
        if fingerprint(example) in seen:
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
                f"{generator.task}: {MAX_REJECTS_PER_EXAMPLE} consecutive rejections at "
                f"{config}. The parameter space is too small for {count} distinct examples; "
                "widen it rather than accepting near-duplicate training data."
            )


def build_eval(
    task: str,
    spec: TaskSpec,
    *,
    size: int = 500,
    seed: int = 7,
    rungs: Optional[Sequence[str]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, List[Dict]], BuildReport]:
    """
    Build a nested eval ladder: one canonical set at the longest rung, shrunk down.

    :param task: Task name.
    :param spec: Its spec.
    :param size: Examples per rung. 500 is the suite floor; anything smaller must be flagged
        inline wherever its numbers are quoted, with an error bar.
    :param seed: Eval seed. Kept distinct from the train seed by default so the two never share a
        stream even if someone passes the same number.
    :param rungs: Rungs to emit; defaults to the task's full ladder.
    :param config: Generator parameter overrides.

    :returns: ``({rung label: examples}, report)``. The report carries the rejection counts,
        which are the guard's output -- a build that discards a third of its draws to contamination
        is not the same build as one that discards none, even though the files look identical.

    :raises ValueError: If ``size`` is below 500, or the generator is not shrink-safe.
    """
    generator = generators.get(task)
    if not generator.shrink_safe:
        raise ValueError(
            f"{task} is not shrink-safe, so a nested ladder would change its gold. Generate each "
            "rung independently and accept the loss of row alignment."
        )
    if size < 500:
        raise ValueError(
            f"eval_size={size} is below the suite floor of 500. A smaller eval inflates noise into "
            "apparent findings; if you need it, build it explicitly and flag the size and its "
            "error bar next to every number."
        )
    labels = list(rungs) if rungs else ladders.rungs_for(task)
    longest = labels[-1]
    resolved = generator.config(**dict(config or {}))
    resolved[generator.scaling_param] = ladders.docs_for_rung(task, longest)

    report = BuildReport(task=task, split="eval")
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


def build_train(
    task: str,
    spec: TaskSpec,
    *,
    total: int = 20_000,
    seed: int = 42,
    rungs: Optional[Sequence[str]] = None,
    config: Optional[Mapping[str, Any]] = None,
    eval_examples: Sequence[Mapping[str, Any]] = (),
) -> Tuple[List[Dict], BuildReport]:
    """
    Build a training set spread uniformly over the rung ladder's document counts.

    Uniform over rungs rather than fixed at one size: a model trained only at the short end has
    never seen the long-context regime it is then measured in, and the ladder is the axis of the
    experiment.

    :param task: Task name.
    :param spec: Its spec.
    :param total: Total examples, split equally across rungs.
    :param seed: Train seed.
    :param rungs: Rungs to cover; defaults to the task's full ladder.
    :param config: Generator parameter overrides.
    :param eval_examples: The eval set, if built. Any train example reusing an eval example's gold
        is rejected -- pass this, or the guard cannot run.

    :returns: ``(examples, report)``, the examples in rung order.
    """
    generator = generators.get(task)
    labels = list(rungs) if rungs else ladders.rungs_for(task)
    per_rung = total // len(labels)
    forbidden = {gold_fingerprint(ex, spec) for ex in eval_examples}

    report = BuildReport(task=task, split="train")
    seen: Set[str] = set()
    examples: List[Dict] = []
    for label in labels:
        resolved = generator.config(**dict(config or {}))
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
            )
        )
        report.counts[label] = len(drawn)
        examples.extend(drawn)
    return examples, report
