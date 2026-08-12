"""Check the ported ``hotpotqa`` contract against the files the shipped numbers were computed on.

Three of this loader's claims are only checkable against the pre-migration build, not against the
dataset card, and each one is silent if wrong:

* the **ladder row** -- 17/36/72/144 documents per rung, which is what ``ctc.data.ladders`` now
  carries instead of BUILD_MATRIX row 2's 11/24/50/100/205;
* the **gold index base** -- 0, shared with the rest of the ``retrieval`` family and opposite to
  contradiction. An off-by-one here scores correct answers at zero, uniformly, and reads as a
  modelling result;
* **two gold per question, flat**, plus hard negatives at ~n/10 and blank titles.

Reads the staged eval ladder on cubbins over ``/net`` (inspection only, a few tens of MB)::

    python debug/hotpotqa_port/verify_shipped_rungs.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

#: Node-local staging on cubbins; ``ctc_suite_staged/eval_rungs/hotpotqa`` is a byte-identical
#: mirror. Read over /net because this is an audit, never from a job -- see local_cluster.md.
STAGED = Path("/net/cubbins/data/prasann/ctc_suite_data/eval_rungs/hotpotqa")

#: rung label -> the document count `ctc.data.ladders.LADDERS["hotpotqa"]` claims for it. 32k is
#: absent from the shipped ladder: it is the fitted extrapolation one rung past what was built.
EXPECTED_DOCS = {"2048": 17, "4096": 36, "8192": 72, "16384": 144}


def check(rung: str, expected: int) -> None:
    """
    :param rung: Rung label, i.e. the token budget in the shipped filename.
    :param expected: Documents per example the ported ladder claims for it.

    :raises AssertionError: If the shipped file disagrees with the ported contract.
    """
    docs: collections.Counter = collections.Counter()
    gold: collections.Counter = collections.Counter()
    hard: collections.Counter = collections.Counter()
    titles = set()
    low, high, rows = 10**9, -1, 0
    with (STAGED / f"rung_{rung}.jsonl").open() as handle:
        for line in handle:
            example = json.loads(line)
            rows += 1
            docs[len(example["documents"])] += 1
            indices = example["gold_doc_indices"]
            assert all(isinstance(g, int) for g in indices), f"rung {rung}: gold is not flat"
            gold[len(indices)] += 1
            low, high = min(low, min(indices)), max(high, max(indices))
            hard[len(example.get("hard_neg_indices") or [])] += 1
            titles.update(d.get("title", "<missing>") for d in example["documents"])

    n_docs = len(example["documents"])
    assert set(docs) == {expected}, f"rung {rung}: {dict(docs)} documents, ladder says {expected}"
    assert set(gold) == {2}, f"rung {rung}: gold counts {dict(gold)}, expected exactly 2"
    # 0-based: the smallest index reached is 0 and the largest is n-1, which no 1-based file can do.
    assert (
        low == 0 and high == n_docs - 1
    ), f"rung {rung}: gold range [{low}, {high}] is not 0-based"
    assert titles == {""}, f"rung {rung}: titles are rendered ({sorted(titles)[:3]})"
    mode = hard.most_common(1)[0][0]
    print(
        f"rung_{rung}: rows={rows} docs={expected} gold=2 base=0 titles=blank "
        f"hard_neg mode={mode} (n/10={expected // 10}) range={min(hard)}-{max(hard)}"
    )


def main() -> None:
    """Check every shipped rung."""
    for rung, expected in EXPECTED_DOCS.items():
        check(rung, expected)


if __name__ == "__main__":
    main()
