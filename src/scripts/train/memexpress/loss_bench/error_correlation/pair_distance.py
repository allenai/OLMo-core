"""
Near-miss analysis for the contradiction tasks: how far off is a wrong predicted pair from the
gold pair it is closest to?

For every INCORRECT contradiction output (``exact_match == 0``) we take each predicted claim-ID
pair, find the gold pair it is nearest to, and record the two element-wise ID distances of that
best alignment, sorted ``(smaller, larger)``, along with their sum (the pair's total displacement). Predicted pairs that exactly hit a gold pair are
dropped -- they are not errors, and including them would put a spike at (0, 0) on top of every
distribution.

Distance of a predicted pair ``p = (a, b)`` to a gold pair ``g = (g1, g2)``::

    d(p, g) = min over the two orientations of  mean(|a - g1|, |b - g2|)

The nearest gold pair is the one minimising ``d``, chosen INDEPENDENTLY for each predicted pair
(no one-to-one assignment -- two predicted pairs may both claim the same gold pair). Ties are
broken on the smaller max-distance, then on gold order.

Worked example from the spec: gold ``[[46, 124], [743, 89], [1, 100]]``, predicted ``[144, 46]``
aligns 46->46 (0) and 144->124 (20), mean 10, which beats every other gold pair, and is recorded
as ``(0, 20)``.

Raw ID distances are NOT comparable across rungs -- claim IDs run 1..100 at the 2k rung and
1..1642 at 32k, so the same distance means very different things. Every distance is therefore also
reported normalised by the example's corpus size (recovered from the highest ``Claim N:`` in the
prompt tail), and the per-rung breakdown is reported alongside the pooled numbers.

A matched CHANCE BASELINE is computed alongside every real number: for each incorrect example,
the model's predicted pairs are replaced by the same number of uniformly random ID pairs drawn from
the same corpus, and scored identically. Without it "the median near-miss is 6% of the corpus away"
is uninterpretable -- with only ~3 gold pairs to be near, random guesses land closer than intuition
suggests, and the whole question is whether the models beat that.

Two sources, same core metric:
  - ``--source local``: the browsable export (``artifact_data.json``), capped at 200 examples per
    (task, source_tag) bucket by ``export_for_artifact.py``. Runs anywhere, no weka.
  - ``--source weka``: every generation dump in ``registry.GENERATION_FILES``. Needs weka mounted
    (run on a CPU gantry job). Uses the scorer's own ``predicted_pairs`` field rather than
    re-parsing the (500-char-truncated) prediction string.

Usage::

    python pair_distance.py --source local --out-dir OUT
    PYTHONPATH=src python pair_distance.py --source weka --out-dir OUT
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CONTRA_TASKS = ("contra", "contra_fever")

# Mirrors evaluate.py's parse_pairs (the scorer's own parser) so the local source, which only has
# the truncated prediction string, splits pairs out of a response the same way the harness did.
_PAIR_RE = re.compile(r"[\[\(]\s*(\d+)\s*,\s*(\d+)\s*[\]\)]")
_CLAIM_RE = re.compile(r"Claim\s+(\d+)\s*:")


def parse_pairs(text: str) -> Optional[List[List[int]]]:
    """Extract integer pairs from a model output. ``None`` means a parse failure."""
    text = text.strip()
    for candidate in [text, re.search(r"\[[\s\S]*\]", text)]:
        if candidate is None:
            continue
        s = candidate if isinstance(candidate, str) else candidate.group()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [
                    sorted([int(p[0]), int(p[1])])
                    for p in parsed
                    if isinstance(p, list) and len(p) == 2
                ]
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    matches = _PAIR_RE.findall(text)
    if matches:
        return [sorted([int(a), int(b)]) for a, b in matches]
    return [] if text in ("[]", "") else None


def corpus_size(prompt_tail: Optional[str], gold: Sequence[Sequence[int]]) -> Optional[int]:
    """Number of claims in the example's corpus.

    The prompt tail is the last 1200 chars of the prompt, which for these tasks always ends with
    the final numbered claims followed by the instruction block, so the largest ``Claim N:`` in it
    is the corpus size. Falls back to the largest gold ID (an underestimate) when the tail is
    missing, and returns None when neither is available so the caller can skip normalising rather
    than silently divide by a wrong number.
    """
    if prompt_tail:
        claims = [int(m) for m in _CLAIM_RE.findall(prompt_tail)]
        if claims:
            return max(claims)
    flat = [i for pair in gold for i in pair]
    return max(flat) if flat else None


def nearest_gold_distances(
    pred: Sequence[int], gold: Sequence[Sequence[int]]
) -> Tuple[int, int, int]:
    """Element distances to the nearest gold pair, as ``(smaller, larger, gold_index)``.

    Both orientations of the predicted pair are tried against each gold pair; the winner minimises
    the mean of the two element distances, then the max, then gold order.
    """
    a, b = int(pred[0]), int(pred[1])
    best: Optional[Tuple[float, int, int, int, int]] = None
    for gi, g in enumerate(gold):
        g1, g2 = int(g[0]), int(g[1])
        for d1, d2 in ((abs(a - g1), abs(b - g2)), (abs(a - g2), abs(b - g1))):
            lo, hi = (d1, d2) if d1 <= d2 else (d2, d1)
            cand = ((lo + hi) / 2.0, hi, gi, lo, hi)
            if best is None or cand < best:
                best = cand
    assert best is not None
    return best[3], best[4], best[2]


class Accumulator:
    """Collects (smaller, larger) distance observations plus the bookkeeping counters."""

    def __init__(self) -> None:
        self.small: List[int] = []
        self.large: List[int] = []
        self.total: List[int] = []
        self.small_norm: List[float] = []
        self.large_norm: List[float] = []
        self.total_norm: List[float] = []
        self.n_examples_incorrect = 0
        self.n_examples_contributing = 0
        self.n_parse_failures = 0
        self.n_empty_predictions = 0
        self.n_exact_hits_dropped = 0

    def add_pair(self, lo: int, hi: int, n_claims: Optional[int]) -> None:
        self.small.append(lo)
        self.large.append(hi)
        self.total.append(lo + hi)
        if n_claims:
            self.small_norm.append(lo / n_claims)
            self.large_norm.append(hi / n_claims)
            self.total_norm.append((lo + hi) / n_claims)

    def summary(self) -> dict:
        def col(vals: List[float], integral: bool) -> dict:
            if not vals:
                return {"n": 0, "mean": None, "median": None, "mode": None, "n_modes": None}
            modes = statistics.multimode(vals) if integral else []
            return {
                "n": len(vals),
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
                "mode": min(modes) if modes else None,
                "n_modes": len(modes) if modes else None,
                "p25": statistics.quantiles(vals, n=4)[0] if len(vals) >= 2 else None,
                "p75": statistics.quantiles(vals, n=4)[2] if len(vals) >= 2 else None,
                "max": max(vals),
            }

        return {
            "n_pairs": len(self.small),
            "n_examples_incorrect": self.n_examples_incorrect,
            "n_examples_contributing": self.n_examples_contributing,
            "n_parse_failures": self.n_parse_failures,
            "n_empty_predictions": self.n_empty_predictions,
            "n_exact_hits_dropped": self.n_exact_hits_dropped,
            "smaller": col(self.small, True),
            "larger": col(self.large, True),
            # Total displacement of the pair: the two element distances added together. Not
            # recoverable from the two columns' summaries, so it is accumulated per observation.
            "total": col(self.total, True),
            "smaller_normalized": col(self.small_norm, False),
            "larger_normalized": col(self.large_norm, False),
            "total_normalized": col(self.total_norm, False),
        }


Record = Tuple[
    str, str, str, str, str, int, list, Optional[float], Optional[str], Optional[list], Optional[str]
]
# (model, task, ladder_version, source_tag, rung, idx, gold, binary, prediction, predicted_pairs,
#  prompt_tail)


def iter_local(path: str) -> Iterable[Record]:
    """Records from the artifact export. Examples are shared across groups, so the caller dedupes."""
    with open(path) as f:
        data = json.load(f)
    for group in data["groups"].values():
        for ex in group.get("examples", []):
            if ex["task"] not in CONTRA_TASKS:
                continue
            for model, mv in ex["models"].items():
                yield (
                    model,
                    ex["task"],
                    ex.get("ladder_version", "?"),
                    ex.get("source_tag", "?"),
                    ex["rung"],
                    ex["idx"],
                    ex["gold"] or [],
                    mv.get("binary"),
                    mv.get("prediction"),
                    None,  # the export keeps only the (truncated) prediction string
                    ex.get("prompt_tail"),
                )


def iter_weka() -> Iterable[Record]:
    """Records straight from the generation dumps, using the scorer's own parsed pairs."""
    from registry import GENERATION_FILES

    for model, entries in GENERATION_FILES.items():
        for task_short, ladder_version, source_tag, path in entries:
            if task_short not in CONTRA_TASKS:
                continue
            if not os.path.exists(path):
                print(f"[{model}] MISSING {path}", flush=True)
                continue
            n = 0
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    detail = rec.get("detail")
                    if detail is None:
                        continue
                    n += 1
                    yield (
                        model,
                        task_short,
                        ladder_version,
                        source_tag,
                        rec["rung"],
                        rec["idx"],
                        detail.get("gold_pairs") or detail.get("gold") or [],
                        detail.get("exact_match"),
                        detail.get("prediction"),
                        detail.get("predicted_pairs"),
                        rec.get("prompt_tail"),
                    )
            print(f"[{model}] {task_short}/{source_tag}/{ladder_version}: {n} rows", flush=True)


def run(records: Iterable[Record], null_trials: int = 20, seed: int = 0) -> dict:
    pooled: Dict[Tuple[str, str], Accumulator] = defaultdict(Accumulator)
    by_rung: Dict[Tuple[str, str, str], Accumulator] = defaultdict(Accumulator)
    null: Dict[Tuple[str, str], Accumulator] = defaultdict(Accumulator)
    rng = random.Random(seed)
    seen: set = set()

    for (
        model,
        task,
        ladder_version,
        source_tag,
        rung,
        idx,
        gold,
        binary,
        prediction,
        predicted_pairs,
        prompt_tail,
    ) in records:
        # The same example is exported under more than one group, so key on its eval coordinates
        # and count it once per model.
        dedupe_key = (model, task, ladder_version, source_tag, rung, idx)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        if binary is None or binary == 1:
            continue  # only incorrect outputs
        if not gold:
            continue  # nothing to measure a near-miss against

        accs = [pooled[(model, task)], by_rung[(model, task, rung)]]
        for acc in accs:
            acc.n_examples_incorrect += 1

        pairs = predicted_pairs
        if pairs is None:
            pairs = parse_pairs(prediction or "")
        if pairs is None:
            for acc in accs:
                acc.n_parse_failures += 1
            continue
        if not pairs:
            for acc in accs:
                acc.n_empty_predictions += 1
            continue

        n_claims = corpus_size(prompt_tail, gold)
        gold_sets = {tuple(sorted(g)) for g in gold}
        contributed = False
        n_real_pairs = 0
        for p in pairs:
            if len(p) != 2:
                continue
            if tuple(sorted(int(x) for x in p)) in gold_sets:
                for acc in accs:
                    acc.n_exact_hits_dropped += 1
                continue
            lo, hi, _ = nearest_gold_distances(p, gold)
            for acc in accs:
                acc.add_pair(lo, hi, n_claims)
            n_real_pairs += 1
            contributed = True
        if contributed:
            for acc in accs:
                acc.n_examples_contributing += 1

        # Matched chance baseline: same example, same corpus, same number of guesses, drawn
        # uniformly instead of predicted. Averaged over several trials to damp sampling noise.
        if n_real_pairs and n_claims and n_claims >= 2:
            nacc = null[(model, task)]
            nacc.n_examples_incorrect += 1
            for _ in range(null_trials):
                for _ in range(n_real_pairs):
                    a, b = rng.sample(range(1, n_claims + 1), 2)
                    if tuple(sorted((a, b))) in gold_sets:
                        nacc.n_exact_hits_dropped += 1
                        continue
                    lo, hi, _gi = nearest_gold_distances((a, b), gold)
                    nacc.add_pair(lo, hi, n_claims)

    out = {
        "pooled": {f"{m}|{t}": a.summary() for (m, t), a in sorted(pooled.items())},
        "by_rung": {f"{m}|{t}|{r}": a.summary() for (m, t, r), a in sorted(by_rung.items())},
        "null_baseline": {f"{m}|{t}": a.summary() for (m, t), a in sorted(null.items())},
        "null_trials": null_trials,
    }
    return out


def _fmt(v: Optional[float], places: int = 1) -> str:
    return "-" if v is None else f"{v:.{places}f}"


def render(out: dict) -> str:
    lines: List[str] = []
    for task in CONTRA_TASKS:
        rows = [(k, v) for k, v in out["pooled"].items() if k.endswith(f"|{task}")]
        if not rows:
            continue
        lines.append(f"\n### {task} (pooled over rungs)\n")
        lines.append(
            "| model | pairs | smaller mean | smaller med | smaller mode | "
            "larger mean | larger med | larger mode | total mean | total med | total mode | "
            "wrong outputs | unparseable |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k, v in rows:
            model = k.split("|")[0]
            s, l, tt = v["smaller"], v["larger"], v["total"]
            unparse = v["n_parse_failures"] + v["n_empty_predictions"]
            frac = unparse / v["n_examples_incorrect"] if v["n_examples_incorrect"] else 0.0
            lines.append(
                f"| {model} | {v['n_pairs']} | {_fmt(s['mean'])} | {_fmt(s['median'])} | "
                f"{_fmt(s['mode'], 0)} | {_fmt(l['mean'])} | {_fmt(l['median'])} | "
                f"{_fmt(l['mode'], 0)} | {_fmt(tt['mean'])} | {_fmt(tt['median'])} | "
                f"{_fmt(tt['mode'], 0)} | {v['n_examples_incorrect']} | "
                f"{unparse} ({frac:.0%}) |"
            )
            nb = out.get("null_baseline", {}).get(k)
            if nb and nb["n_pairs"]:
                ns, nl, nt = nb["smaller"], nb["larger"], nb["total"]
                lines.append(
                    f"| _{model} (chance)_ | {nb['n_pairs']} | {_fmt(ns['mean'])} | "
                    f"{_fmt(ns['median'])} | {_fmt(ns['mode'], 0)} | {_fmt(nl['mean'])} | "
                    f"{_fmt(nl['median'])} | {_fmt(nl['mode'], 0)} | {_fmt(nt['mean'])} | "
                    f"{_fmt(nt['median'])} | {_fmt(nt['mode'], 0)} | - | - |"
                )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["local", "weka"], default="local")
    ap.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifact_data.json"),
        help="artifact_data.json path (--source local only)",
    )
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--tag", default=None, help="suffix for the output filenames")
    ap.add_argument(
        "--null-trials",
        type=int,
        default=20,
        help="random-guess trials per incorrect example for the matched chance baseline",
    )
    args = ap.parse_args()

    records = iter_local(args.data) if args.source == "local" else iter_weka()
    out = run(records, null_trials=args.null_trials)
    out["_meta"] = {"source": args.source, "tasks": list(CONTRA_TASKS)}

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or args.source
    json_path = os.path.join(args.out_dir, f"pair_distance_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    table = render(out)
    md_path = os.path.join(args.out_dir, f"pair_distance_{tag}.md")
    with open(md_path, "w") as f:
        f.write(f"# Near-miss distances ({tag})\n{table}\n")
    print(table)
    print(f"\n[done] wrote {json_path} and {md_path}", flush=True)


if __name__ == "__main__":
    main()
