"""
Measure tokens-per-document for the absence / xabsence ladders against real staged data.

``ctc.data.ladders`` rows are only meaningful once *one* answer to "how long is this example?"
is chosen, and the pre-migration tree had four. This uses the same one contradiction's measured
row used: render the real prompt through the task spec, count with a Qwen3 tokenizer.

The substrate is the pre-migration suite's own staged eval files, so the numbers describe the
corpus the published rungs were computed against rather than a fresh snapshot. Both ladders are
fitted as ``tokens ~= intercept + slope * n_docs`` and then inverted per rung, which is how
``hotpotqa``'s row was recalibrated. The ``absence_pubmed`` rows are measured too, but only as
context: BUILD_MATRIX row 18 replaced that variant with the Gutenberg one on 2026-07-19.

Run (login node, tokenizer read from the scratch HF cache)::

    PYTHONPATH=ctc/src python debug/absence_port/measure_ladders.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

TOKENIZER = (
    "/scratch/users/prasann/huggingface-cache/hub/models--Qwen--Qwen3-4B-Base/"
    "snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
STAGED = Path("/scratch/users/prasann/corpus-reasoning/data")

#: ``(fit group, task spec, staged file)``. Only the ``absence`` and ``xabsence`` groups are fitted
#: into a ladder; ``absence_pubmed`` is measured for comparison and not used.
CASES = [
    ("absence", "absence", "absence_eval_gutenberg_n10_k3.jsonl"),
    ("absence", "absence", "absence_eval_gutenberg_n50_k3.jsonl"),
    ("absence", "absence", "absence_eval_gutenberg_n200_k3.jsonl"),
    ("absence_pubmed", "absence", "absence_eval_pubmed_n20_p01.jsonl"),
    ("absence_pubmed", "absence", "absence_eval_pubmed_n100_p01.jsonl"),
    ("absence_pubmed", "absence", "absence_eval_pubmed_n250_p01.jsonl"),
    ("xabsence", "xabsence", "xabsence_eval_pubmed_p8_k3.jsonl"),
    ("xabsence", "xabsence", "xabsence_eval_pubmed_p18_k3.jsonl"),
    ("xabsence", "xabsence", "xabsence_eval_pubmed_p48_k3.jsonl"),
]

#: The rung labels every CTC ladder carries, with their token budgets.
RUNGS = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768}


def fit(points: Sequence[Tuple[int, float]]) -> Tuple[float, float]:
    """
    Least-squares fit of ``tokens = intercept + slope * n_docs``.

    An intercept rather than a bare tok/doc ratio: the instruction, the header and (for absence)
    the ``Second version:`` framing are a fixed cost that a pure ratio charges to the short rungs,
    which is exactly how a ladder ends up undershooting its own labels at the short end.

    :param points: ``(n_docs, median tokens)`` observations.

    :returns: ``(intercept, slope)``.
    """
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    var = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / var
    return mean_y - slope * mean_x, slope


def main() -> int:
    from transformers import AutoTokenizer

    from ctc.format import registry
    from ctc.tasks import load_all

    load_all()
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    observed: Dict[str, List[Tuple[int, float]]] = {}

    print(f"{'file':<44} {'n_docs':>7} {'median tok':>11} {'tok/doc':>8}")
    for group, task, name in CASES:
        path = STAGED / name
        if not path.exists():
            print(f"{name:<44} MISSING")
            continue
        spec = registry.get(task)
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()][:40]
        lengths: List[int] = [len(tok(spec.build_prompt(r))["input_ids"]) for r in rows]
        n_docs = int(statistics.median(len(r["documents"]) for r in rows))
        median = statistics.median(lengths)
        observed.setdefault(group, []).append((n_docs, median))
        print(f"{name:<44} {n_docs:>7} {median:>11.0f} {median / n_docs:>8.1f}")

    for group, points in observed.items():
        if len(points) < 2:
            continue
        intercept, slope = fit(points)
        ladder = {
            label: max(1, round((budget - intercept) / slope)) for label, budget in RUNGS.items()
        }
        print(f"\n{group}: tokens ~= {intercept:.1f} + {slope:.2f} * n_docs")
        print(f"  fitted ladder: {ladder}")
        for n_docs, median in points:
            print(f"  check n={n_docs}: fit {intercept + slope * n_docs:.0f} vs measured {median}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
