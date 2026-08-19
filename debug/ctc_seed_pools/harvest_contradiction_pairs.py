"""Recover a reusable ``pairs_path`` file for ``contradiction`` from the shipped 20k train build.

Contradiction gold is ``(S, S')`` where ``S`` is a real PubMed sentence and ``S'`` is a
model-written sentence that cannot also be true. Mining costs an LLM serving run, and the
pre-migration tree already paid for it: the audited 20k joint-uniform train file
(``contradiction_train_pubmed_realistic_n50-950_k3.jsonl``, cubbins, 2026-07-19) holds 60k gold
pairs built with ``--seed 42``. The shipped rows keep only shuffled documents plus 1-based
``gold_doc_indices``, so which half is real, and its abstract id, are recovered the same way
``harvest_redundancy_pairs.py`` recovered them: an index over EVERY PubMedQA abstract's sentences,
keyed by dataset row index -- the exact ``abstract_id`` ``load_abstracts`` uses, so an id recovered
here means the same thing at build time (fillers may not come from an abstract that contributed
gold, or a filler can restate the fact the contradiction denies).

Pairs where both or neither side is found in PubMed are dropped rather than guessed; the counts
are printed so the loss is visible.

Writes ``contradiction_pairs.jsonl`` next to this script.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ctc" / "src"))

from ctc.data.sources.pubmed import split_sentences  # noqa: E402

SHIPPED = Path(
    "/net/cubbins/data/prasann/ctc_suite_data/contradiction_pool/"
    "contradiction_train_pubmed_realistic_n50-950_k3.jsonl"
)
OUT = Path(__file__).with_name("contradiction_pairs.jsonl")


def main() -> None:
    from datasets import load_dataset

    started = time.time()
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    where = {}
    for i in range(len(dataset)):
        if i in (0, 1, 4, 9, 9999, 99999):
            print(f"  indexing abstract {i + 1:,}/{len(dataset):,} ({time.time() - started:.0f}s)")
        for chunk in dataset[i]["context"]["contexts"]:
            if chunk:
                for sentence in split_sentences(chunk):
                    where.setdefault(sentence, str(i))
    print(f"indexed {len(where):,} sentences over {len(dataset):,} abstracts")

    pairs = {}
    stats = {"rows": 0, "pairs_seen": 0, "both_real": 0, "neither_real": 0}
    with SHIPPED.open(encoding="utf-8") as handle:
        for n, line in enumerate(handle):
            if n in (0, 1, 4, 9, 99, 999) or n % 2000 == 0:
                print(f"  row {n:,} ({time.time() - started:.0f}s, {len(pairs):,} pairs)")
            row = json.loads(line)
            stats["rows"] += 1
            docs = [d["text"] for d in row["documents"]]
            for a, b in row["gold_doc_indices"]:
                stats["pairs_seen"] += 1
                sa, sb = docs[a - 1], docs[b - 1]
                in_a, in_b = sa in where, sb in where
                if in_a and in_b:
                    stats["both_real"] += 1
                    continue  # cannot tell which half the model wrote; drop rather than guess
                if not in_a and not in_b:
                    stats["neither_real"] += 1
                    continue
                claim, contradiction = (sa, sb) if in_a else (sb, sa)
                pairs.setdefault(
                    (claim, contradiction),
                    {
                        "claim": claim,
                        "contradiction": contradiction,
                        "abstract_id": where[claim],
                        "mode": "realistic",
                    },
                )
    stats["recovered"] = len(pairs)

    with OUT.open("w", encoding="utf-8") as handle:
        for entry in pairs.values():
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
