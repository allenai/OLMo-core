"""
Measure prompt length per document for reorder / qdmatch / grouping_labeled.

Same method the ported ``absence``/``xabsence`` rows used: render the SHIPPED pre-migration eval
files through this repo's own ``spec.build_prompt``, tokenize with the Qwen3 tokenizer, fit
``tokens ~= a + b * n_docs`` over the available document counts, and solve for each rung's budget.
Rendering through the real prompt builder is what makes the fit usable -- the per-document token
cost is only part of it; the instruction header and the per-item template are the rest.

Run:  PYTHONPATH=ctc/src python debug/ctc_three_task_port/measure_ladders.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

DATA = Path("/accounts/projects/berkeleynlp/prasann/projects/corpus-reasoning/data")

#: task -> [(documents per example, shipped file)]
FILES = {
    "reorder": [
        (5, "reorder_gutenberg100w_n5_eval_500.jsonl"),
        (20, "reorder_gutenberg100w_n20_eval_500.jsonl"),
        (50, "reorder_gutenberg100w_n50_eval_500.jsonl"),
    ],
    "qdmatch": [
        (40, "qdmatch_eval_nq_q20_n20_k3_separate.jsonl"),
        (100, "qdmatch_eval_nq_q50_n50_k3_separate.jsonl"),
        (200, "qdmatch_eval_nq_q100_n100_k3_separate.jsonl"),
        (500, "qdmatch_eval_nq_q250_n250_k3_separate.jsonl"),
    ],
    "grouping_labeled": [
        (20, "openalex_grouping_n20_levels_eval_500.jsonl"),
        (100, "openalex_grouping_n100_levels_eval_200.jsonl"),
    ],
}

RUNGS = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768}
SAMPLE = 40


#: Qwen3-4B-Base's tokenizer files, copied off horton's HF cache so the measurement runs offline
#: and reproducibly. Same vocab as every other Qwen3 rung calibration in ``ctc.data.ladders``.
TOKENIZER = Path(__file__).parent / "tok_qwen3_4b_base"


def tokenizer():
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TOKENIZER))


def reorder_normalised(tok) -> None:
    """
    Re-measure ``reorder`` with each passage's internal whitespace collapsed.

    The shipped files carry Gutenberg's own hard line wraps inside every passage, and those
    newlines are real tokens. The ported generator builds its passages out of
    :mod:`ctc.data.sources.gutenberg`'s prose runs, which are whitespace-normalised, so the shipped
    fit would overstate its per-passage cost. Same words, same word-count target -- only the wraps
    differ, which is exactly what this isolates.
    """
    import re

    from ctc.format import registry

    spec = registry.get("reorder")
    points = []
    for n_docs, name in FILES["reorder"]:
        rows = []
        with (DATA / name).open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= SAMPLE:
                    break
                row = json.loads(line)
                row["documents"] = [
                    {"text": re.sub(r"\s+", " ", d["text"]).strip()} for d in row["documents"]
                ]
                rows.append(row)
        lengths = [len(tok.encode(spec.build_prompt(row))) for row in rows]
        targets = [len(tok.encode(json.dumps(row["gold_order"]))) for row in rows]
        median = statistics.median(lengths)
        points.append((n_docs, median))
        print(
            f"reorder (normalised) n={n_docs:<4d} median={median:8.0f} tokens "
            f"({median / n_docs:.1f}/doc), target median={statistics.median(targets):.0f}"
        )
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum(
        (x - mean_x) ** 2 for x in xs
    )
    intercept = mean_y - slope * mean_x
    print(f"  fit: tokens ~= {intercept:.1f} + {slope:.2f} * n_docs")
    print(
        "  ladder:",
        {label: max(1, round((b - intercept) / slope)) for label, b in RUNGS.items()},
    )


def main() -> int:
    from ctc.format import registry
    from ctc.tasks import load_all

    load_all()
    tok = tokenizer()
    for task, entries in FILES.items():
        spec = registry.get(task)
        points = []
        for n_docs, name in entries:
            path = DATA / name
            if not path.exists():
                print(f"  MISSING {path}", file=sys.stderr)
                continue
            rows = []
            with path.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= SAMPLE:
                        break
                    rows.append(json.loads(line))
            lengths = [len(tok.encode(spec.build_prompt(row))) for row in rows]
            median = statistics.median(lengths)
            points.append((n_docs, median))
            print(
                f"{task:17s} n={n_docs:<4d} median={median:8.0f} tokens  ({median / n_docs:.1f}/doc)"
            )
        if len(points) < 2:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
        var = sum((x - mean_x) ** 2 for x in xs)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var
        intercept = mean_y - slope * mean_x
        print(f"  fit: tokens ~= {intercept:.1f} + {slope:.2f} * n_docs")
        ladder = {
            label: max(1, round((budget - intercept) / slope)) for label, budget in RUNGS.items()
        }
        print(f"  ladder: {ladder}")
        print()
    reorder_normalised(tok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
