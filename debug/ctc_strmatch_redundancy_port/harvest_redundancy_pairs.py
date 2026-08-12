"""Recover a reusable ``pairs_path`` file for ``redundancy`` from the shipped pre-migration JSONL.

Redundancy gold is ``(S, S')`` where ``S`` is a real PubMed sentence and ``S'`` is an LLM
paraphrase, and the hard negatives are same-abstract sentence pairs an LLM judge cleared as NOT
redundant. Both halves cost a serving run to mine, and the pre-migration tree already paid for it:
``/accounts/projects/berkeleynlp/prasann/projects/corpus-reasoning/data/redundancy_*.jsonl`` holds
2000 train + 244 eval examples built with ``--pool-abstracts 20000 --seed 42``.

The shipped rows keep only the shuffled documents plus 1-based ``gold_doc_indices`` /
``_hardneg_pairs``, so two things have to be recovered:

* **which half of a gold pair is the real sentence** -- the one that appears in the PubMed pool;
  the other is the paraphrase;
* **the abstract id**, which is load-bearing (fillers may not come from an abstract that
  contributed gold to the same example, or a filler can restate the fact the gold pair states).

Both come from an index over **every** PubMedQA abstract, keyed by dataset row index -- which is
exactly what ``load_abstracts`` uses as its ``abstract_id``, so an id recovered here means the same
thing at build time. The whole dataset rather than a 20k sample because the shipped files were
plainly built from a larger pool than the generator's ``--pool-abstracts 20000`` default: with a
20k sample only half the gold pairs resolve, and spot-checking three of the unresolved ones found
all three in the full dataset at rows 185298 / 123279 / 16414.

Writes ``redundancy_pairs.jsonl`` next to this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ctc" / "src"))

from ctc.data.sources.pubmed import split_sentences  # noqa: E402

SHIPPED = Path("/accounts/projects/berkeleynlp/prasann/projects/corpus-reasoning/data")
FILES = [
    "redundancy_train_pubmed_both_n100_k3_hn6.jsonl",
    "redundancy_eval_pubmed_both_n100_k3_hn6.jsonl",
    "redundancy_train_pubmed_both_n20_k2_hn4.jsonl",
    "redundancy_eval_pubmed_both_n20_k2_hn4.jsonl",
]
OUT = Path(__file__).with_name("redundancy_pairs.jsonl")


def main() -> None:
    from datasets import load_dataset

    dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    where = {}
    for i in range(len(dataset)):
        for chunk in dataset[i]["context"]["contexts"]:
            if chunk:
                for sentence in split_sentences(chunk):
                    where.setdefault(sentence, str(i))
    print(f"indexed {len(where):,} sentences over {len(dataset):,} abstracts")

    gold, hardneg = {}, {}
    stats = {"rows": 0, "gold": 0, "both_real": 0, "neither_real": 0, "hn": 0, "hn_split": 0}
    for name in FILES:
        path = SHIPPED / name
        if not path.exists():
            print(f"  MISSING {path}")
            continue
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            stats["rows"] += 1
            docs = [d["text"] for d in row["documents"]]
            for a, b in row["gold_doc_indices"]:
                sa, sb = docs[a - 1], docs[b - 1]
                in_a, in_b = sa in where, sb in where
                if in_a and in_b:
                    stats["both_real"] += 1
                    continue  # cannot tell which half the model wrote; drop rather than guess
                if not in_a and not in_b:
                    stats["neither_real"] += 1
                    continue
                claim, paraphrase = (sa, sb) if in_a else (sb, sa)
                gold.setdefault(
                    (claim, paraphrase),
                    {
                        "claim": claim,
                        "paraphrase": paraphrase,
                        "abstract_id": where[claim],
                        "source_file": name,
                    },
                )
            for a, b in row.get("_hardneg_pairs") or []:
                sa, sb = docs[a - 1], docs[b - 1]
                aid_a, aid_b = where.get(sa), where.get(sb)
                if aid_a is None or aid_b is None:
                    continue
                if aid_a != aid_b:
                    stats["hn_split"] += 1
                    continue  # the pre-migration pool only ever drew SAME-abstract hard negatives
                hardneg.setdefault(
                    (sa, sb), {"a": sa, "b": sb, "abstract_id": aid_a, "source_file": name}
                )
    stats["gold"], stats["hn"] = len(gold), len(hardneg)

    with OUT.open("w", encoding="utf-8") as handle:
        for row in gold.values():
            handle.write(json.dumps({"kind": "gold", **row}, ensure_ascii=True) + "\n")
        for row in hardneg.values():
            handle.write(json.dumps({"kind": "hardneg", **row}, ensure_ascii=True) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
