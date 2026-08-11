"""
Integrity and shortcut checks over built data. A build stage, not after-the-fact debugging.

Every check here corresponds to a defect that reached published numbers and was first mistaken for
a modelling result, and they come in two kinds.

**Integrity** is the familiar kind: schema violations, a ladder whose rungs grade different
questions, train and eval sharing examples.

**Shortcuts** are the kind this suite actually got bitten by. The data was correctly split, and gold
still carried a signature a trivial heuristic could read off
(``records/ctc-setting-verification-2026-07-23.md``): cycle's gold entities appeared exactly twice
while background frequency grew with N, so gold was "the rarest names" and the shortcut got
*stronger* as N grew; groups4's distractors were all more than X from everything, so one close pair
identified gold without finding a G-clique at all. Both fixes live in the generators; these probes
are how we learn if an edit undoes one.

The probes are cheap heuristics, not proofs. A probe near chance is evidence the obvious shortcut is
absent, not that the task is hard.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from ..format.registry import TaskSpec
from .build import fingerprint, gold_fingerprint
from .schema import validate_all

__all__ = ["AuditResult", "ProbeResult", "audit", "run_probes"]


@dataclass
class AuditResult:
    """
    :param checks: check name -> one-line outcome, always reported.
    :param problems: Fatal findings; a build with any of these must not be staged.
    :param warnings: Worth reading, but not invalidating.
    """

    checks: Dict[str, str] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """:returns: True when nothing fatal was found."""
        return not self.problems

    def absorb(self, other: "AuditResult") -> None:
        """
        :param other: A sub-check's result to merge in.
        """
        self.checks.update(other.checks)
        self.problems.extend(other.problems)
        self.warnings.extend(other.warnings)

    def report(self) -> str:
        """:returns: A printable summary."""
        return "\n".join(
            [f"  {name}: {outcome}" for name, outcome in self.checks.items()]
            + [f"  WARN  {w}" for w in self.warnings]
            + [f"  FAIL  {p}" for p in self.problems]
        )


@dataclass(frozen=True)
class ProbeResult:
    """
    :param name: Probe name.
    :param score: Heuristic hit rate in ``[0, 1]`` -- roughly "how often this alone finds gold".
    :param ceiling: Above this, the shortcut is strong enough to explain the task.
    :param detail: One line of context.
    """

    name: str
    score: float
    ceiling: float
    detail: str = ""

    @property
    def failed(self) -> bool:
        """:returns: True when the probe fired."""
        return self.score > self.ceiling

    def __str__(self) -> str:
        mark = "FAIL" if self.failed else "ok"
        tail = f" — {self.detail}" if self.detail else ""
        return f"[{mark}] {self.name}: {self.score:.3f} (ceiling {self.ceiling:.3f}){tail}"


def _gold_positions(example: Mapping, spec: TaskSpec) -> List[int]:
    """0-based positions of the gold documents, whatever base the task declares."""
    flat: List[int] = []
    for entry in example.get("gold_doc_indices") or []:
        flat.extend(entry) if isinstance(entry, (list, tuple)) else flat.append(entry)
    return [p - spec.gold_index_base for p in flat]


# ── shortcut probes ─────────────────────────────────────────────────────────────────────────────


def gold_position_bias(examples: Sequence[Mapping], spec: TaskSpec) -> ProbeResult:
    """
    Does gold sit at predictable positions? Measured as the largest share in any one decile.

    "Gold is near the end" is learnable without reading anything, and end-of-context placement was
    separately measured to move a task's score by ~0.22 in the shared-corpus experiments.

    :param examples: Examples to probe.
    :param spec: Their task spec.

    :returns: The probe result. Chance is 0.1.
    """
    deciles: Counter = Counter()
    for example in examples:
        n = max(1, len(example["documents"]))
        deciles.update(min(9, 10 * p // n) for p in _gold_positions(example, spec))
    total = sum(deciles.values())
    if not total:
        return ProbeResult("gold_position_bias", 0.0, 0.25, "no gold indices")
    top, count = deciles.most_common(1)[0]
    return ProbeResult("gold_position_bias", count / total, 0.25, f"heaviest decile #{top}")


def gold_length_bias(examples: Sequence[Mapping], spec: TaskSpec) -> ProbeResult:
    """
    Can gold be found by document length alone? Scored as how often a gold document is the longest
    or shortest in its corpus.

    :param examples: Examples to probe.
    :param spec: Their task spec.

    :returns: The probe result.
    """
    hits = total = 0
    for example in examples:
        lengths = [len(d["text"]) for d in example["documents"]]
        gold = set(_gold_positions(example, spec))
        if not gold or len(lengths) < 3:
            continue
        hits += bool(gold & {lengths.index(max(lengths)), lengths.index(min(lengths))})
        total += 1
    score = hits / total if total else 0.0
    return ProbeResult("gold_length_bias", score, 0.5, f"{hits}/{total} examples")


def cycle_frequency_gap(examples: Sequence[Mapping], spec: TaskSpec) -> ProbeResult:
    """
    Are the cycle's entities identifiable as the rarest names?

    The exact statistic behind the 2026-07-23 diagnosis: gold entities had one in-edge and one
    out-edge regardless of N, while background degree grew with N.

    :param examples: cycle examples.
    :param spec: The cycle spec.

    :returns: The probe result -- the mean share of gold entities landing among the k rarest, where
        k is the number of gold entities.
    """
    scores = []
    for example in examples:
        entities = [
            [t.split()[0], t.split()[-1]] for t in (d["text"] for d in example["documents"])
        ]
        gold = _gold_positions(example, spec)
        if not gold:
            continue
        freq: Counter = Counter(e for pair in entities for e in pair)
        gold_entities = {e for i in gold for e in entities[i]}
        rarest = {e for e, _ in sorted(freq.items(), key=lambda kv: kv[1])[: len(gold_entities)]}
        scores.append(len(gold_entities & rarest) / max(1, len(gold_entities)))
    score = sum(scores) / len(scores) if scores else 0.0
    return ProbeResult("cycle_frequency_gap", score, 0.5, "gold recoverable as the rarest names")


def closest_pair_is_gold(examples: Sequence[Mapping], spec: TaskSpec) -> ProbeResult:
    """
    For the numeric tasks: is "the two closest values" enough to find gold?

    Reads the values the generator recorded in ``_meta``, so this measures the construction rather
    than an arithmetic parser's accuracy.

    :param examples: mathmatch, groups4 or textgroups examples.
    :param spec: Their spec.

    :returns: The probe result.
    """
    hits = total = 0
    for example in examples:
        meta = example.get("_meta") or {}
        values = meta.get("answer_values") or meta.get("counts")
        gold = set(_gold_positions(example, spec))
        if not values or not gold or len(values) < 3:
            continue
        i, j = min(
            combinations(range(len(values)), 2), key=lambda p: abs(values[p[0]] - values[p[1]])
        )
        hits += {i, j} <= gold
        total += 1
    score = hits / total if total else 0.0
    return ProbeResult("closest_pair_is_gold", score, 0.5, f"{hits}/{total} examples")


Probe = Callable[[Sequence[Mapping], TaskSpec], ProbeResult]

#: task -> extra probes beyond the generic two.
PROBES: Dict[str, List[Probe]] = {
    "cycle": [cycle_frequency_gap],
    "groups4": [closest_pair_is_gold],
    "mathmatch": [closest_pair_is_gold],
    "textgroups": [closest_pair_is_gold],
}

#: Probes whose high score is a known property of the construction, not a defect. Recorded so a
#: settled result is not rediscovered as a finding on every build.
ACCEPTED: Dict[str, Dict[str, str]] = {
    "mathmatch": {
        "closest_pair_is_gold": (
            "ACCEPTED: mathmatch places every distractor more than X from everything, so at K=1 the "
            "closest pair is gold by construction. Its difficulty comes from N and from evaluating "
            "the arithmetic; groups4 is the variant that closes this deliberately."
        )
    }
}


def run_probes(task: str, examples: Sequence[Mapping], spec: TaskSpec) -> List[ProbeResult]:
    """
    Run the generic probes plus any registered for this task.

    :param task: Task name.
    :param examples: A sample of its examples; the pair probes are O(N^2) in documents.
    :param spec: The task spec.

    :returns: One result per probe, with accepted ones downgraded so they cannot fail the build.
    """
    results = [gold_position_bias(examples, spec), gold_length_bias(examples, spec)]
    results += [probe(examples, spec) for probe in PROBES.get(task, [])]
    accepted = ACCEPTED.get(task, {})
    return [
        ProbeResult(r.name, r.score, 1.01, accepted[r.name]) if r.name in accepted else r
        for r in results
    ]


# ── integrity checks ────────────────────────────────────────────────────────────────────────────


def check_split_separation(
    train: Sequence[Mapping], evalset: Sequence[Mapping], spec: TaskSpec
) -> AuditResult:
    """
    Train and eval must share no example, and no *question*.

    Two levels, because they fail differently. A shared full example is plain duplication. A shared
    **gold fingerprint** -- the same claim pair or cycle with different filler around it -- is the
    subtler one: the corpora differ, every surface check passes, and the model has still been
    trained on the eval question.

    :param train: Training examples.
    :param evalset: Eval examples across every rung.
    :param spec: Their task spec.

    :returns: The audit result.
    """
    result = AuditResult()
    train_full = {fingerprint(e) for e in train}
    shared_full = train_full & {fingerprint(e) for e in evalset}
    eval_gold = {gold_fingerprint(e, spec) for e in evalset}
    shared_gold = {gold_fingerprint(e, spec) for e in train} & eval_gold

    result.checks["exact overlap"] = f"{len(shared_full)} shared example(s)"
    result.checks["gold overlap"] = (
        f"{len(shared_gold)} of {len(eval_gold)} distinct eval questions also in train"
    )
    if shared_full:
        result.problems.append(f"{len(shared_full)} example(s) appear in both train and eval")
    if shared_gold:
        result.problems.append(
            f"{len(shared_gold)} eval question(s) ({len(shared_gold) / max(1, len(eval_gold)):.1%}) "
            "also appear in train with different filler -- same gold, so differing corpora do not "
            "make them different questions"
        )
    if len(train) != len(train_full):
        result.warnings.append(f"{len(train) - len(train_full)} duplicate example(s) within train")
    return result


def check_ladder_nesting(rungs: Mapping[str, Sequence[Mapping]], spec: TaskSpec) -> AuditResult:
    """
    Every rung must grade the same questions over nested corpora.

    Without this the ladder's x-axis is not a controlled variable: "the context got longer" is
    confounded with "the questions changed", which is the entire comparison these experiments make.

    :param rungs: rung label -> examples, ascending by length.
    :param spec: The task spec.

    :returns: The audit result.
    """
    result = AuditResult()
    labels = list(rungs)
    if len(labels) < 2:
        result.checks["ladder"] = "single rung; nothing to compare"
        return result

    sizes = {label: len(rows) for label, rows in rungs.items()}
    if len(set(sizes.values())) != 1:
        result.problems.append(f"rungs have different row counts: {sizes}")
        return result
    result.checks["row alignment"] = f"{sizes[labels[0]]} rows at every rung"

    mismatched = not_nested = 0
    for row in range(sizes[labels[0]]):
        mismatched += len({gold_fingerprint(rungs[la][row], spec) for la in labels}) > 1
        for shorter, longer in zip(labels, labels[1:]):
            short = {d["text"] for d in rungs[shorter][row]["documents"]}
            long = {d["text"] for d in rungs[longer][row]["documents"]}
            not_nested += not short <= long
    result.checks["gold identical across rungs"] = f"{mismatched} mismatched row(s)"
    result.checks["distractor nesting"] = f"{not_nested} non-nested pair(s)"
    if mismatched:
        result.problems.append(
            f"{mismatched} row(s) have different gold at different rungs, so the rungs are not "
            "measuring the same questions"
        )
    if not_nested:
        result.problems.append(
            f"{not_nested} rung pair(s) are not nested: a shorter rung holds documents the longer "
            "one does not"
        )
    for label, rows in rungs.items():
        if len(rows) < 500:
            result.warnings.append(
                f"rung {label} has eval_size={len(rows)}, below the suite floor of 500: quote the "
                "size and its error bar inline next to every number"
            )
    return result


def audit(
    task: str,
    spec: TaskSpec,
    *,
    train: Sequence[Mapping] = (),
    rungs: Optional[Mapping[str, Sequence[Mapping]]] = None,
    probe_sample: int = 200,
) -> AuditResult:
    """
    Run every applicable check over one task's built data.

    :param task: Task name.
    :param spec: Its spec.
    :param train: Training examples, if built.
    :param rungs: rung label -> eval examples, if built.
    :param probe_sample: Eval examples to run the shortcut probes over; capped because the pair
        probes are O(N^2) in documents.

    :returns: The combined result.
    """
    rungs = dict(rungs or {})
    result = AuditResult()

    for label, rows in (("train", list(train)), *rungs.items()):
        if not rows:
            continue
        problems = validate_all(rows, spec, require_gold=task != "oolong")
        result.checks[f"schema/{label}"] = f"{len(rows)} row(s), {len(problems)} invalid"
        result.problems.extend(f"{label}: {p}" for p in problems[:5])

    if rungs:
        result.absorb(check_ladder_nesting(rungs, spec))

    flat_eval = [ex for rows in rungs.values() for ex in rows]
    if train and flat_eval:
        result.absorb(check_split_separation(train, flat_eval, spec))

    for probe in run_probes(task, (flat_eval or list(train))[:probe_sample], spec):
        result.checks[f"shortcut/{probe.name}"] = str(probe)
        if probe.failed:
            result.problems.append(
                f"shortcut probe {probe.name} scored {probe.score:.3f} against a ceiling of "
                f"{probe.ceiling:.3f}: gold may be findable without doing the task"
            )
    return result
