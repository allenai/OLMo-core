"""Fit the ``strmatch`` and ``redundancy`` rung ladders against the real Qwen3 tokenizer.

``BUILD_MATRIX.md`` row 20 gives strmatch ``38/82/170/350/700`` and flags itself
``~45 (synth x1.5-3) ... calibrate before freezing n values``; row 17 (redundancy) was struck out
entirely, so it has no rung ladder at all. Both rows land in :data:`ctc.data.ladders.LADDERS` as
*measurements* rather than estimates, which is what ``CALIBRATION`` is for -- the contradiction
ladder was wrong by ~1.8x for exactly this reason.

Renders real examples through each task's own prompt assembler, so what is measured is the string
the model sees, not a proxy.

Usage::

    python debug/ctc_strmatch_redundancy_port/calibrate_ladders.py strmatch
    python debug/ctc_strmatch_redundancy_port/calibrate_ladders.py redundancy
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ctc" / "src"))

from ctc.data.generators import base as generators  # noqa: E402
from ctc.format import registry  # noqa: E402
from ctc.tasks import load_all  # noqa: E402

RUNGS = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768}
PROBE_SIZES = {"strmatch": (30, 60, 120, 240), "redundancy": (40, 80, 160, 320)}
SAMPLES = 12


def main() -> None:
    task = sys.argv[1]
    pairs_path = sys.argv[2] if len(sys.argv) > 2 else None
    load_all()
    spec = registry.get(task)
    generator = generators.get(task)
    corpus = None
    if generator.corpus is not None:
        corpus = generator.load_corpus(pairs_path=pairs_path)
        print(f"pool: {len(corpus)} gold pairs, {len(corpus.hardnegs)} hard negatives")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    points = []
    for n_docs in PROBE_SIZES[task]:
        lengths = []
        for i in range(SAMPLES):
            kwargs = dict(generator.config(), num_docs=n_docs)
            if generator.indexed:
                kwargs["index"] = i
            if corpus is not None:
                kwargs["corpus"] = corpus
            example = generator.build_example(random.Random(f"cal:{n_docs}:{i}"), **kwargs)
            if example is None:
                continue
            lengths.append(len(tokenizer(spec.build_prompt(example))["input_ids"]))
        median = statistics.median(lengths)
        points.append((n_docs, median))
        print(f"  n={n_docs:5d}  median tokens {median:8.0f}  ({median / n_docs:6.2f} tok/doc)")

    # Least squares over (n, tokens): the intercept is the instruction plus the numbering scaffold,
    # which matters at the short rungs and is why a bare tok/doc ratio undershoots there.
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    print(f"\n  tokens ~= {intercept:.1f} + {slope:.3f} * n_docs")
    row = {label: max(1, round((budget - intercept) / slope)) for label, budget in RUNGS.items()}
    print(f"  {task}: {row}")


if __name__ == "__main__":
    main()
