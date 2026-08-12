"""Calibrate the hotpotqa rung ladder: documents per rung -> median rendered prompt tokens.

Measures the REAL thing: a pool from the real corpus, examples through the real generator, prompts
through `retrieval.build_prompt`, tokens through the Qwen3 tokenizer. `ctc.data.ladders.CALIBRATION`
distinguishes rows measured this way from rows estimated offline, and the contradiction ladder is
why -- its counts were fit against the wrong filler pool and overshot every rung by ~1.8x.

The CE pass is off here: it reorders hard negatives and cannot change any document's length, so it
has no effect on the calibration and would cost a GPU.

    python debug/hotpotqa_port/measure_ladder.py
"""

from __future__ import annotations

import random
import statistics

from transformers import AutoTokenizer

from ctc.data.generators import base as generators
from ctc.data.sources import hotpotqa
from ctc.format import registry
from ctc.tasks import load_all

RUNGS = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768}
SAMPLE = 60


def median_tokens(gen, spec, pool, tokenizer, num_docs: int) -> float:
    """:returns: Median rendered prompt length, in tokens, at ``num_docs`` documents."""
    lengths = []
    for index in range(SAMPLE):
        example = gen.build_example(
            random.Random(index), index=index, corpus=pool, **gen.config(num_docs=num_docs)
        )
        if example is None:
            continue
        lengths.append(len(tokenizer(spec.build_prompt(example))["input_ids"]))
    return statistics.median(lengths)


def main() -> None:
    load_all()
    gen = generators.get("hotpotqa")
    spec = registry.get("retrieval")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    pool = hotpotqa.load_pool(num_questions=1500, filler_pool=12_000, ce_filter=False)
    print(f"pool: {len(pool.queries)} queries, {len(pool.corpus)} filler docs")
    gold = [len(q.gold) for q in pool.queries]
    hard = [len(q.hard) for q in pool.queries]
    print(
        f"gold/query: min {min(gold)} max {max(gold)} | hard/query: median {statistics.median(hard)}"
    )

    per_doc = median_tokens(gen, spec, pool, tokenizer, 42) - median_tokens(
        gen, spec, pool, tokenizer, 2
    )
    print(f"~{per_doc / 40:.1f} tokens per document (marginal, 2 -> 42 docs)\n")

    for label, budget in RUNGS.items():
        low, high, best = 2, 400, 2
        while low <= high:  # binary search the largest count that still fits the budget
            mid = (low + high) // 2
            if median_tokens(gen, spec, pool, tokenizer, mid) <= budget:
                best, low = mid, mid + 1
            else:
                high = mid - 1
        print(
            f"{label}: {best} docs -> {median_tokens(gen, spec, pool, tokenizer, best):.0f} tokens"
        )


if __name__ == "__main__":
    main()
