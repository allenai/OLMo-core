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

import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set

from ..format import metrics
from ..format.registry import TaskSpec
from .build import fingerprint, gold_fingerprint
from .generators import base as generators
from .schema import validate_all

__all__ = [
    "AuditResult",
    "ProbeResult",
    "audit",
    "run_probes",
    "check_rung_sizes",
    "unmatched_by_lexical_overlap",
]


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


#: How far above chance a probe must score before it is a finding. A shortcut that beats chance by
#: less than this is not worth a build failure.
MARGIN = 0.20

#: Below this many probed examples a firing probe is reported but cannot fail the build. The
#: probes are hit rates; at n=10 a single unlucky draw moves one by 0.1, MARGIN is barely one
#: binomial standard error, and the false alarm teaches everyone to pass --force -- which is
#: worse than having no probe. The floor keeps every >=500-example ladder exactly as strict
#: as before; it exists for --allow-small-eval demo builds.
MIN_PROBE_SAMPLES = 50


@dataclass(frozen=True)
class ProbeResult:
    """
    A probe's hit rate, **against its own chance baseline**.

    The baseline is not decoration. Several of these heuristics are near-certain to hit by
    accident when gold is a large fraction of the corpus: at the textgroups 2k rung, 6 of 11
    documents are gold, so "is a gold document the longest or the shortest?" is true ~82% of the
    time in data with no shortcut at all. A fixed ceiling reports that as a failure, blocks a
    legitimate build, and teaches everyone to pass ``--force`` -- which is worse than having no
    probe. So each probe computes what it would score on shortcut-free data of the same shape, and
    only a margin above *that* counts.

    :param name: Probe name.
    :param score: Heuristic hit rate in ``[0, 1]`` -- roughly "how often this alone finds gold".
    :param chance: What this probe scores on data of the same shape with no shortcut.
    :param detail: One line of context.
    :param ceiling: Overrides ``chance + MARGIN``. Used to mark a probe as accepted.
    """

    name: str
    score: float
    chance: float
    detail: str = ""
    ceiling: Optional[float] = None

    @property
    def threshold(self) -> float:
        """:returns: The score above which this is a finding."""
        return self.ceiling if self.ceiling is not None else min(1.0, self.chance + MARGIN)

    @property
    def failed(self) -> bool:
        """:returns: True when the probe beat chance by more than :data:`MARGIN`."""
        return self.score > self.threshold

    def __str__(self) -> str:
        mark = "FAIL" if self.failed else "ok"
        tail = f" — {self.detail}" if self.detail else ""
        return (
            f"[{mark}] {self.name}: {self.score:.3f} vs chance {self.chance:.3f} "
            f"(threshold {self.threshold:.3f}){tail}"
        )


def _gold_positions(example: Mapping, spec: TaskSpec) -> List[int]:
    """0-based positions of the gold documents, whatever base the task declares."""
    flat: List[int] = []
    for entry in example.get("gold_doc_indices") or []:
        flat.extend(entry) if isinstance(entry, (list, tuple)) else flat.append(entry)
    return [p - spec.gold_index_base for p in flat]


# ── shortcut probes ─────────────────────────────────────────────────────────────────────────────


def _max_decile_chance(total: int) -> float:
    """
    :param total: Observations spread over the ten deciles.

    :returns: What the *maximum* decile share scores when the observations are uniform. Uniform
        data puts a tenth in each bin, but the max over ten noisy bins sits above 0.1, and more so
        with few observations -- so a probe scored as "the heaviest decile" must be compared
        against the expected maximum, not against the mean.
    """
    return min(1.0, 0.1 + 0.9 * (10 / total) ** 0.5) if total else 0.1


def gold_position_bias(examples: Sequence[Mapping], spec: TaskSpec) -> ProbeResult:
    """
    Does gold sit at predictable positions? Measured as the largest share in any one decile.

    "Gold is near the end" is learnable without reading anything, and end-of-context placement was
    separately measured to move a task's score by ~0.22 in the shared-corpus experiments.

    :param examples: Examples to probe.
    :param spec: Their task spec.

    :returns: The probe result, against :func:`_max_decile_chance`'s baseline.
    """
    deciles: Counter = Counter()
    for example in examples:
        n = max(1, len(example["documents"]))
        deciles.update(min(9, 10 * p // n) for p in _gold_positions(example, spec))
    total = sum(deciles.values())
    if not total:
        return ProbeResult("gold_position_bias", 0.0, 0.1, "no gold indices")
    top, count = deciles.most_common(1)[0]
    return ProbeResult(
        "gold_position_bias", count / total, _max_decile_chance(total), f"heaviest decile #{top}"
    )


def gold_length_bias(examples: Sequence[Mapping], spec: TaskSpec) -> ProbeResult:
    """
    Can gold be found by document length alone? Scored as how often a gold document is the longest
    or shortest in its corpus.

    :param examples: Examples to probe.
    :param spec: Their task spec.

    :returns: The probe result.
    """
    hits = total = 0
    chance_sum = 0.0
    for example in examples:
        lengths = [len(d["text"]) for d in example["documents"]]
        gold = set(_gold_positions(example, spec))
        n, g = len(lengths), len(gold)
        if not gold or n < 3:
            continue
        hits += bool(gold & {lengths.index(max(lengths)), lengths.index(min(lengths))})
        total += 1
        # P(at least one of the two extreme positions is gold) if gold were placed at random:
        # 1 - P(both extremes are non-gold).
        chance_sum += 1.0 - ((n - g) / n) * ((n - g - 1) / (n - 1))
    score = hits / total if total else 0.0
    chance = chance_sum / total if total else 0.0
    return ProbeResult("gold_length_bias", score, chance, f"{hits}/{total} examples")


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
    scores: List[float] = []
    chances: List[float] = []
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
        chances.append(len(gold_entities) / max(1, len(freq)))
    score = sum(scores) / len(scores) if scores else 0.0
    # Picking the k rarest entities at random recovers k/|entities| of the gold ones.
    chance = sum(chances) / len(chances) if chances else 0.0
    return ProbeResult("cycle_frequency_gap", score, chance, "gold recoverable as the rarest names")


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
    chance_sum = 0.0
    for example in examples:
        meta = example.get("_meta") or {}
        values = meta.get("answer_values") or meta.get("counts")
        gold = set(_gold_positions(example, spec))
        if not values or not gold or len(values) < 3:
            continue
        # The closest pair of scalars is always adjacent in value order, so sorting replaces the
        # C(N,2) enumeration -- hundreds of millions of materialised tuples at the ultra-long
        # rungs, which dominated the whole audit. Sorting by (value, index) and breaking distance
        # ties on the smaller (i, j) reproduces the enumeration's first-minimum exactly.
        order = sorted(range(len(values)), key=lambda i: (values[i], i))
        i, j = min(
            ((min(a, b), max(a, b)) for a, b in zip(order, order[1:])),
            key=lambda p: (abs(values[p[0]] - values[p[1]]), p),
        )
        hits += {i, j} <= gold
        total += 1
        # A random pair being gold-contained -- the rate this probe would hit with no signal:
        # C(gold, 2) / C(N, 2), which needs no enumeration either.
        n, g = len(values), len(gold)
        chance_sum += (g * (g - 1)) / (n * (n - 1))
    score = hits / total if total else 0.0
    chance = chance_sum / total if total else 0.0
    return ProbeResult("closest_pair_is_gold", score, chance, f"{hits}/{total} examples")


#: Above this many documents the O(N^2) cross-corpus probe is skipped for that example. The probe
#: is a heuristic over the *construction*, which does not change with the rung, so measuring it on
#: the short rungs answers the question at a hundredth of the cost.
PAIR_PROBE_MAX_DOCS = 300


def _tokens(text: str) -> Set[str]:
    """:param text: A document body. :returns: Its lowercased alphanumeric token set."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def unmatched_by_lexical_overlap(examples: Sequence[Mapping], spec: TaskSpec) -> ProbeResult:
    """
    For ``xabsence``: can the unmatched claims be found by word overlap alone?

    The task is supposed to require recognising a *paraphrase* across the corpus boundary. Score
    every document by its best word-Jaccard against any document in the other corpus, take the ``k``
    lowest, and ask how many of them are gold. A near-copy twin -- or the ``exact`` pool mode, whose
    twin is byte-identical -- makes this 1.0, and the task is then a string match dressed up as a
    semantic one.

    :param examples: xabsence examples.
    :param spec: The xabsence spec.

    :returns: The probe result. Chance is ``k/n``: the share of gold recovered by naming ``k``
        documents at random.
    """
    scores: List[float] = []
    chances: List[float] = []
    probed = 0
    for example in examples:
        docs = example["documents"]
        gold = set(_gold_positions(example, spec))
        if not gold or len(docs) < 3 or len(docs) > PAIR_PROBE_MAX_DOCS:
            continue
        probed += 1
        sides = [doc.get("corpus", "A") for doc in docs]
        tokens = [_tokens(doc["text"]) for doc in docs]
        best: List[float] = []
        for i, side in enumerate(sides):
            others = [j for j in range(len(docs)) if sides[j] != side]
            best.append(
                max(
                    (
                        len(tokens[i] & tokens[j]) / max(1, len(tokens[i] | tokens[j]))
                        for j in others
                    ),
                    default=0.0,
                )
            )
        lowest = sorted(range(len(docs)), key=lambda i: best[i])[: len(gold)]
        scores.append(len(gold & set(lowest)) / len(gold))
        chances.append(len(gold) / len(docs))
    score = sum(scores) / len(scores) if scores else 0.0
    chance = sum(chances) / len(chances) if chances else 0.0
    return ProbeResult(
        "unmatched_by_lexical_overlap",
        score,
        chance,
        f"{probed} example(s) at <= {PAIR_PROBE_MAX_DOCS} documents",
    )


def overlap_pair_is_gold(examples: Sequence[Mapping], spec: TaskSpec) -> ProbeResult:
    """
    For the pair family: does ranking pairs by shared-word count alone find gold?

    ``strmatch`` and ``redundancy`` both ask "which pairs of items stand in relation R?", and for
    both there is a heuristic that never reads a word's position or meaning: score every pair by
    how many words the two share, take the top one. This measures how often that pair is gold.

    It is the same statistic that caught ``cycle``'s rarest-names shortcut, and it fires for two
    very different reasons, which is why it is registered for both tasks and accepted for only one:

    * on the pre-migration ``strmatch`` it scored **1.000** (200/200 on the shipped
      ``rung_2048.jsonl``, chance 0.004) because gold shared exactly ``span_len`` words and the
      hard negatives exactly one fewer -- the contiguity half of the criterion was decorative, and
      the construction was fixed;
    * on ``redundancy`` it scores well above chance because a paraphrase shares its original's
      content words. That is the criterion showing through a blunt proxy, not an alternative route
      to it, so it is recorded in :data:`ACCEPTED` with its number rather than fixed away.

    :param examples: Pair-task examples.
    :param spec: Their task spec.

    :returns: The probe result. Chance is the mean share of pairs that are gold, i.e. what naming
        one pair at random would score.
    """
    hits = probed = 0
    chance_sum = 0.0
    for example in examples:
        docs = example["documents"]
        gold = {tuple(sorted(pair)) for pair in (example.get("gold_doc_indices") or [])}
        if not gold or len(docs) < 3 or len(docs) > PAIR_PROBE_MAX_DOCS:
            continue
        probed += 1
        tokens = [_tokens(doc["text"]) for doc in docs]
        pairs = list(combinations(range(len(docs)), 2))
        best = max(pairs, key=lambda pair: len(tokens[pair[0]] & tokens[pair[1]]))
        hits += tuple(sorted(p + spec.gold_index_base for p in best)) in gold
        chance_sum += len(gold) / len(pairs)
    score = hits / probed if probed else 0.0
    chance = chance_sum / probed if probed else 0.0
    return ProbeResult(
        "overlap_pair_is_gold",
        score,
        chance,
        f"{hits}/{probed} example(s) at <= {PAIR_PROBE_MAX_DOCS} documents",
    )


def reorder_display_order_leak(examples: Sequence[Mapping], spec: TaskSpec) -> ProbeResult:
    """
    For ``reorder``: does the order the passages are *shown* in already carry the answer?

    The whole task rests on one ``rng.shuffle``. A shuffle that is partial, biased, or accidentally
    applied to the wrong list leaves display order correlated with source order, and a model that
    answers ``[1, 2, 3, ...]`` without reading a word scores well on the suite's only Kendall-tau
    metric. That failure would be invisible to every other check here: the data validates, the
    permutation is a permutation, and only its *distribution* is wrong.

    :param examples: reorder examples.
    :param spec: The reorder spec.

    :returns: The probe result. Score is the mean Kendall tau between ``gold_order`` and the
        identity ordering, rescaled from ``[-1, 1]`` to ``[0, 1]``; chance is 0.5, since a uniform
        permutation has expected tau 0. Passing the default margin means tau stayed below 0.4.
    """
    taus = []
    for example in examples:
        gold_order = example.get("gold_order") or []
        if len(gold_order) < 3:
            continue
        taus.append(metrics.kendall_tau(list(range(1, len(gold_order) + 1)), list(gold_order)))
    score = (sum(taus) / len(taus) + 1) / 2 if taus else 0.5
    return ProbeResult(
        "reorder_display_order_leak",
        score,
        0.5,
        f"mean tau {2 * score - 1:+.3f} over {len(taus)} example(s)",
    )


def reorder_length_position_bias(examples: Sequence[Mapping], spec: TaskSpec) -> ProbeResult:
    """
    For ``reorder``: is a passage's length a clue to where it belongs?

    The specific hazard is the trailing partial passage. Group consecutive sentences up to a word
    target and the last group of every run is a short remainder; emit it and "the shortest passage
    goes last" pins one position of the permutation for free, worth more Kendall tau the shorter
    the rung. :func:`ctc.tasks.reorder.generate.passage_runs` drops it, and this is the check that
    it still does.

    :param examples: reorder examples.
    :param spec: The reorder spec.

    :returns: The probe result -- the heaviest decile of *source* position for the shortest passage
        in each example, against the expected-maximum baseline for uniform data.
    """
    deciles: Counter = Counter()
    for example in examples:
        gold_order = example.get("gold_order") or []
        docs = example["documents"]
        if len(gold_order) != len(docs) or len(docs) < 3:
            continue
        # gold_order[source position] = display id, so this inverts it to lengths in source order.
        lengths = [len(docs[display - 1]["text"]) for display in gold_order]
        deciles[min(9, 10 * lengths.index(min(lengths)) // len(lengths))] += 1
    total = sum(deciles.values())
    if not total:
        return ProbeResult("reorder_length_position_bias", 0.0, 0.1, "no gold_order")
    top, count = deciles.most_common(1)[0]
    return ProbeResult(
        "reorder_length_position_bias",
        count / total,
        _max_decile_chance(total),
        f"shortest passage sits in source decile #{top} most often",
    )


def qd_pair_position_bias(examples: Sequence[Mapping], spec: TaskSpec) -> ProbeResult:
    """
    For ``qdmatch``: does a gold document sit at a predictable place in the document block?

    The generic :func:`gold_position_bias` cannot see this task at all -- it reads
    ``gold_doc_indices``, and qdmatch's gold is ``gold_pairs``. This measures the same property
    over the right field, and *within the document block* rather than over the whole item list:
    under the ``separate`` layout every document is in the back half by construction, so measuring
    over all items would report a 0.2 concentration that means nothing.

    :param examples: qdmatch examples.
    :param spec: The qdmatch spec.

    :returns: The probe result -- the heaviest decile of a gold document's rank among the document
        items, against the expected-maximum baseline for uniform data.
    """
    deciles: Counter = Counter()
    for example in examples:
        docs = example["documents"]
        positions = [i for i, item in enumerate(docs) if item.get("type") != "query"]
        rank = {position: i for i, position in enumerate(positions)}
        if len(positions) < 3:
            continue
        for pair in example.get("gold_pairs") or []:
            index = pair[1] - spec.gold_index_base
            if index in rank:
                deciles[min(9, 10 * rank[index] // len(positions))] += 1
    total = sum(deciles.values())
    if not total:
        return ProbeResult("qd_pair_position_bias", 0.0, 0.1, "no gold pairs")
    top, count = deciles.most_common(1)[0]
    return ProbeResult(
        "qd_pair_position_bias",
        count / total,
        _max_decile_chance(total),
        f"heaviest document-block decile #{top}",
    )


Probe = Callable[[Sequence[Mapping], TaskSpec], ProbeResult]

#: task -> extra probes beyond the generic two.
PROBES: Dict[str, List[Probe]] = {
    "cycle": [cycle_frequency_gap],
    "groups4": [closest_pair_is_gold],
    "mathmatch": [closest_pair_is_gold],
    "textgroups": [closest_pair_is_gold],
    "strmatch": [overlap_pair_is_gold],
    "redundancy": [overlap_pair_is_gold],
    "xabsence": [unmatched_by_lexical_overlap],
    # Keyed by LADDER name, like PROBES' other entries and like GENERATORS: the two qdmatch
    # corpora are separate rows and either could regress on its own.
    "reorder": [reorder_display_order_leak, reorder_length_position_bias],
    "qdmatch_nq": [qd_pair_position_bias],
    "qdmatch_hpqa": [qd_pair_position_bias],
}

#: Probes whose high score is a known property of the construction, not a defect. Recorded so a
#: settled result is not rediscovered as a finding on every build.
ACCEPTED: Dict[str, Dict[str, str]] = {
    "redundancy": {
        "overlap_pair_is_gold": (
            "ACCEPTED: redundancy's gold IS a paraphrase pair, so the two sentences share content "
            "words by definition and a bag-of-words ranking is a blunt version of the criterion "
            "rather than a way around it. The construction bounds it as far as it can without an "
            "LLM in the loop -- H=2K planted same-abstract non-redundant pairs, each the "
            "highest-overlap pair its abstract offers, and gold above max_overlap word-Jaccard "
            "dropped as a near duplicate -- and the residual is real, measured at matched size: "
            "this probe scores 0.500 on the shipped redundancy_eval_pubmed_both_n100_k3_hn6.jsonl "
            "and 0.400 on this construction, both at n=100 against a 0.001 chance baseline. By "
            "word-Jaccard instead of raw shared-word count the shipped file scores 0.610, and a "
            "top-3 Jaccard ranking recovers 52% of its gold pairs. Quote that alongside any "
            "redundancy result; do not read the ACCEPTED tag as the shortcut having been ruled "
            "out."
        )
    },
    "mathmatch": {
        "closest_pair_is_gold": (
            "ACCEPTED: mathmatch places every distractor more than X from everything, so at K=1 the "
            "closest pair is gold by construction. Its difficulty comes from N and from evaluating "
            "the arithmetic; groups4 is the variant that closes this deliberately."
        )
    },
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
    out: List[ProbeResult] = []
    for r in results:
        if r.name in accepted:
            r = ProbeResult(r.name, r.score, r.chance, accepted[r.name], ceiling=1.01)
        elif r.failed and len(examples) < MIN_PROBE_SAMPLES:
            r = ProbeResult(
                r.name,
                r.score,
                r.chance,
                f"{r.detail}; ADVISORY: fired, but {len(examples)} example(s) is below the "
                f"{MIN_PROBE_SAMPLES}-sample floor, so it cannot fail the build -- rebuild "
                "larger before trusting either the data or this probe",
                ceiling=1.01,
            )
        out.append(r)
    return out


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


def check_rung_sizes(rungs: Mapping[str, Sequence[Mapping]]) -> AuditResult:
    """
    The size floor, for ladders that cannot be checked for nesting.

    :param rungs: rung label -> examples.

    :returns: The audit result. Only the eval_size warning, which applies to every ladder however
        it was constructed.
    """
    result = AuditResult()
    for label, rows in rungs.items():
        if len(rows) < 500:
            result.warnings.append(
                f"rung {label} has eval_size={len(rows)}, below the suite floor of 500: quote the "
                "size and its error bar inline next to every number"
            )
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
    nested: Optional[bool] = None,
) -> AuditResult:
    """
    Run every applicable check over one task's built data.

    :param task: Ladder name.
    :param spec: Its grading spec.
    :param train: Training examples, if built.
    :param rungs: rung label -> eval examples, if built.
    :param probe_sample: Eval examples to run the shortcut probes over; capped because the pair
        probes are O(N^2) in documents.
    :param nested: Whether this ladder's rungs are supposed to grade the same questions. Defaults
        to asking the generator. ``oolong`` recomputes its gold per draw and so cannot be nested;
        reporting that as a defect on every build would train everyone to ignore the finding, and
        an ignored audit is the same as no audit.

    :returns: The combined result.
    """
    rungs = dict(rungs or {})
    result = AuditResult()
    if nested is None:
        try:
            nested = generators.get(task).nested_ladder
        except (KeyError, AttributeError, ValueError):
            nested = True

    for label, rows in (("train", list(train)), *rungs.items()):
        if not rows:
            continue
        problems = validate_all(rows, spec, require_gold=task != "oolong")
        result.checks[f"schema/{label}"] = f"{len(rows)} row(s), {len(problems)} invalid"
        result.problems.extend(f"{label}: {p}" for p in problems[:5])

    if rungs and nested:
        result.absorb(check_ladder_nesting(rungs, spec))
    elif rungs:
        result.absorb(check_rung_sizes(rungs))
        result.checks["ladder"] = (
            "rungs built independently by design: they do NOT grade the same questions, so "
            "rung-to-rung deltas include eval-set resampling noise"
        )

    flat_eval = [ex for rows in rungs.values() for ex in rows]
    if train and flat_eval:
        result.absorb(check_split_separation(train, flat_eval, spec))

    for probe in run_probes(task, (flat_eval or list(train))[:probe_sample], spec):
        result.checks[f"shortcut/{probe.name}"] = str(probe)
        if probe.failed:
            result.problems.append(
                f"shortcut probe {probe.name} scored {probe.score:.3f} against a chance baseline "
                f"of {probe.chance:.3f}: gold may be findable without doing the task"
            )
    return result
