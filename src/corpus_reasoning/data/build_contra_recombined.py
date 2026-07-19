"""
Build a large contradiction dataset by **recombining** existing LLM-authored contradiction pairs.

⚠⚠ **DO NOT USE THIS AS-IS TO BUILD TRAINING DATA. Its filler sourcing destroys the task's hard
negatives.** Measured 2026-07-16 (see ``records/contradiction-data-and-base-hygiene.md`` §3b): the
output trains a model to **f1 0.585 under FULL attention**, where the same recipe on the original
generator's examples scores ~0.93 on a harder corpus. Cause: real examples co-sample distractors *with*
the gold pair from a related pool, so the nearest non-gold pair is nearly as similar as the gold pair
(0.372 vs 0.445) and the model must actually detect contradiction. Filling from a global pool drops the
nearest distractor to **0.163** against a gold pair at 0.461, so "pick the two most similar claims"
solves training — a shortcut that collapses on the topically-coherent eval. Difficulty is a property of
the JOINT (gold, distractor) sample and cannot be reconstructed by recombination.

⚠ Note that every validation this script's output passed — zero eval contamination, zero pair reuse,
gold-pair distances matched to the eval, no duplicate docs, no filler that is secretly gold. The
symptom was only visible as **normal train CE (0.0144) with far-below-expected eval f1**. Checks
verified the properties we thought of; none asked whether the task was still the same task.

To scale the example count, use ``generate_pubmed_contradiction_data.py --expand-from-train``, which
preserves each example's own distractor context (1 source example -> 1 output example). The
**decontamination logic below is still correct and worth reusing** (the 14 eval-poisoned pairs are
real); the filler sourcing is what must change -- a fixed version must draw distractors so that the
best non-gold similarity matches the eval's ~0.37, not a uniform global sample.

Why this exists: :mod:`generate_pubmed_contradiction_data` authors contradiction pairs with an LLM
(gemini), so scaling the example count normally costs API calls. Its ``--expand-from-train`` mode
skips the LLM but is **1 source example -> 1 output example**, so it cannot exceed the source size
(2000). This script instead pools every pair the LLM has *already* authored across all existing
contradiction files and recombines them into fresh examples: new partners, new distractors, new
positions. No LLM calls.

The pair pool is large enough to build a **zero-reuse** dataset (each contradiction pair used in
exactly one example), which removes any pair-memorization confound.

Output is layer-1 **unified JSONL** (``documents`` / ``queries`` / ``answers`` /
``gold_doc_indices`` / ``source``), the same schema
:mod:`generate_pubmed_contradiction_data` emits -- feed it to
``src/scripts/data/convert_unified_to_document_landmark.py``. For contradiction, ``queries`` and
``answers`` are empty: the prompt is rendered from ``documents`` and the answer is derived from
``gold_doc_indices`` (see ``olmo_core.data.corpus_reasoning_prompts``).

Design decisions worth knowing (see ``records/multihop-gold-routing-experiment.md``):

* **Style matching.** ``--mode both`` (what the eval sets use) is 50/50 simple+subtle, so the
  ``simple`` and ``subtle`` pools are the same two style families and may be pooled with ``both``.
  ``realistic`` is a *different* perturbation style ("fully rephrases -- no near-duplicate tells")
  and is **excluded** by default.
* **Eval decontamination.** 14 train pairs appear verbatim in the eval sets. Every pair appearing in
  any eval file is dropped. This is not optional.
* **Filler safety.** A distractor that happens to be one half of *another* example's contradiction
  pair could silently form an unlabeled positive with its partner. Every text that participates in
  any gold pair anywhere is therefore excluded from the filler pool.
* **Placement.** Gold positions are sampled to match the eval's gold-pair distance distribution. No
  minimum distance is imposed -- ~4% of eval pairs are adjacent, and the eval set is fixed, so a
  minimum would create a train/eval mismatch on pairs the model is actually tested on.

Example::

    python src/corpus_reasoning/data/build_contra_recombined.py \\
      --data-dir /scratch/users/prasann/corpus-reasoning/data \\
      --num-docs 50 --pairs-per-example 3 \\
      --out /scratch/users/prasann/corpus-reasoning/data/contradiction_train_pubmed_recomb_n50_k3.jsonl
"""

import argparse
import glob
import json
import os
import random
from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple

Pair = Tuple[str, str]

# `both` = 50/50 simple+subtle, so these three are one style family. `realistic` is a different
# perturbation style and would shift the distribution away from the `both` eval sets.
STYLE_MATCHED_MODES = ("both", "simple", "subtle")


def _docs_of(record: dict) -> List[str]:
    """Return document texts, tolerating both the ``{"text": ...}`` and bare-string layouts."""
    return [d["text"] if isinstance(d, dict) else d for d in (record.get("documents") or [])]


def _pairs_of(record: dict) -> List[Pair]:
    """Extract normalized (text_a, text_b) gold pairs from a unified record (1-indexed)."""
    docs = _docs_of(record)
    out: List[Pair] = []
    for p in record.get("gold_doc_indices") or []:
        if not (isinstance(p, list) and len(p) == 2):
            continue
        try:
            a, b = docs[p[0] - 1].strip(), docs[p[1] - 1].strip()
        except (IndexError, AttributeError):
            continue
        out.append(tuple(sorted((a, b))))  # type: ignore[arg-type]
    return out


def _iter_records(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _mode_of(path: str) -> str:
    """`contradiction_train_pubmed_<mode>_n50_k3.jsonl` -> `<mode>`."""
    base = os.path.basename(path)
    return base.split("pubmed_")[1].split("_")[0] if "pubmed_" in base else ""


def collect(data_dir: str, modes: Sequence[str]) -> Tuple[Dict[Pair, str], Set[Pair], List[str]]:
    """
    Scan every contradiction file once.

    :returns: ``(train_pairs, eval_pairs, filler_texts)`` where ``train_pairs`` maps each pair to the
        style mode it came from, ``eval_pairs`` is every pair appearing in any eval file, and
        ``filler_texts`` are candidate distractors (deduped, gold-participating texts NOT yet
        removed -- the caller does that once the full gold set is known).
    """
    train_pairs: Dict[Pair, str] = {}
    eval_pairs: Set[Pair] = set()
    filler: Set[str] = set()

    for path in sorted(glob.glob(os.path.join(data_dir, "contradiction_*pubmed*.jsonl"))):
        is_eval = "_eval_" in os.path.basename(path)
        mode = _mode_of(path)
        for rec in _iter_records(path):
            pairs = _pairs_of(rec)
            if is_eval:
                eval_pairs.update(pairs)
                continue
            if mode not in modes:
                continue
            for p in pairs:
                train_pairs.setdefault(p, mode)
            gold_here = {t for p in pairs for t in p}
            for t in _docs_of(rec):
                t = t.strip()
                if t and t not in gold_here:
                    filler.add(t)
    return train_pairs, eval_pairs, sorted(filler)


def place(n_docs: int, n_pairs: int, rng: random.Random) -> List[Tuple[int, int]]:
    """
    Choose ``n_pairs`` disjoint position-pairs among ``n_docs`` slots, uniformly at random.

    Uniform sampling of distinct positions reproduces the eval's gold-pair distance distribution
    (mean ~16.7 at n=50) as a matter of course, since the eval was generated the same way. No minimum
    distance is imposed -- see the module docstring.

    :returns: ``n_pairs`` disjoint ``(lo, hi)`` 0-indexed position pairs, ``lo < hi``.
    """
    slots = rng.sample(range(n_docs), 2 * n_pairs)
    rng.shuffle(slots)
    out = []
    for i in range(n_pairs):
        a, b = slots[2 * i], slots[2 * i + 1]
        out.append((min(a, b), max(a, b)))
    return out


def build(args) -> None:
    rng = random.Random(args.seed)

    train_pairs, eval_pairs, filler_all = collect(args.data_dir, args.modes)
    print(f"scanned {args.data_dir}")
    print(f"  train pairs (style-matched {list(args.modes)}): {len(train_pairs)}")
    print(f"  eval pairs (all eval files):                    {len(eval_pairs)}")

    # --- decontaminate: drop any train pair that appears verbatim in an eval set.
    contaminated = [p for p in train_pairs if p in eval_pairs]
    for p in contaminated:
        del train_pairs[p]
    print(f"  DROPPED {len(contaminated)} eval-contaminated train pairs")

    # --- filler pool: exclude every text participating in ANY gold pair (train or eval), so a
    # distractor can never be one half of some other example's contradiction.
    gold_texts = {t for p in list(train_pairs) + list(eval_pairs) for t in p}
    filler = [t for t in filler_all if t not in gold_texts]
    print(f"  filler pool: {len(filler)} (from {len(filler_all)} before gold exclusion)")

    pool = sorted(train_pairs)
    rng.shuffle(pool)
    max_examples = len(pool) // args.pairs_per_example
    n_examples = max_examples if args.num_examples <= 0 else min(args.num_examples, max_examples)
    n_filler = args.num_docs - 2 * args.pairs_per_example
    if n_filler < 0:
        raise ValueError(f"--num-docs {args.num_docs} too small for {args.pairs_per_example} pairs")
    if len(filler) < n_filler:
        raise ValueError(f"filler pool {len(filler)} < {n_filler} needed per example")

    print(
        f"\nbuilding {n_examples} examples "
        f"(max zero-reuse = {max_examples}), n_docs={args.num_docs}, "
        f"{args.pairs_per_example} pairs + {n_filler} fillers each"
    )

    style_used: Counter = Counter()
    written = 0
    with open(args.out, "w") as f:
        for i in range(n_examples):
            chosen = pool[i * args.pairs_per_example : (i + 1) * args.pairs_per_example]
            for p in chosen:
                style_used[train_pairs[p]] += 1

            docs: List[str] = [""] * args.num_docs
            positions = place(args.num_docs, args.pairs_per_example, rng)
            gold_doc_indices = []
            for (lo, hi), pair in zip(positions, chosen):
                a, b = pair
                if rng.random() < 0.5:  # don't let claim order encode pair order
                    a, b = b, a
                docs[lo], docs[hi] = a, b
                gold_doc_indices.append([lo + 1, hi + 1])  # unified format is 1-indexed

            free = [j for j in range(args.num_docs) if not docs[j]]
            for j, t in zip(free, rng.sample(filler, len(free))):
                docs[j] = t

            f.write(
                json.dumps(
                    {
                        "documents": [{"text": t} for t in docs],
                        "queries": [],
                        "answers": [],
                        "gold_doc_indices": sorted(gold_doc_indices),
                        "source": "pubmed_recombined",
                    }
                )
                + "\n"
            )
            written += 1

    print(f"\nwrote {written} examples -> {args.out}")
    tot = sum(style_used.values())
    print(
        "  gold pair style mix: "
        + ", ".join(f"{m} {c} ({c/tot:.0%})" for m, c in style_used.most_common())
    )
    print(f"  pair reuse: every pair used exactly once ({tot} pair-slots, {tot} unique pairs)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-dir",
        default="/scratch/users/prasann/corpus-reasoning/data",
        help="dir holding the existing contradiction_*pubmed*.jsonl files",
    )
    p.add_argument("--out", required=True, help="output unified JSONL path")
    p.add_argument("--num-docs", type=int, default=50, help="corpus size n (docs per example)")
    p.add_argument("--pairs-per-example", type=int, default=3, help="k = gold contradiction pairs")
    p.add_argument(
        "--num-examples",
        type=int,
        default=0,
        help="0 = as many as the pool allows at zero pair reuse",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=list(STYLE_MATCHED_MODES),
        help="perturbation styles to pool. Default is the eval-matched family; adding "
        "'realistic' shifts the distribution away from the `both` eval sets.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    main()
