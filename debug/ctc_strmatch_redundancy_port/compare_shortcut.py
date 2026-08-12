"""Measure ``overlap_pair_is_gold`` on the SHIPPED pre-migration data and on the ported build.

The probe is only worth its build-time cost if the two numbers differ, so this runs it on matched
corpus sizes: shipped ``strmatch/rung_2048.jsonl`` (n=38) against a ported build at n=38, and the
shipped ``redundancy_eval_pubmed_both_n100_k3_hn6.jsonl`` (n=100) against a ported build at n=100.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ctc" / "src"))

from ctc.data import audit  # noqa: E402
from ctc.data.generators import base as generators  # noqa: E402
from ctc.format import registry  # noqa: E402
from ctc.tasks import load_all  # noqa: E402

SHIPPED = {
    "strmatch": (
        "/scratch/users/prasann/ctc_suite_staged/eval_rungs/strmatch/rung_2048.jsonl",
        38,
    ),
    "redundancy": (
        "/accounts/projects/berkeleynlp/prasann/projects/corpus-reasoning/data/"
        "redundancy_eval_pubmed_both_n100_k3_hn6.jsonl",
        100,
    ),
}
PAIRS = str(Path(__file__).with_name("redundancy_pairs.jsonl"))
SAMPLE = 100


def main() -> None:
    load_all()
    for task, (path, n_docs) in SHIPPED.items():
        spec = registry.get(task)
        with open(path, encoding="utf-8") as handle:
            shipped = [json.loads(line) for _, line in zip(range(SAMPLE), handle)]
        print(f"{task} shipped  n={n_docs}: {audit.overlap_pair_is_gold(shipped, spec)}")

        generator = generators.get(task)
        corpus = generator.load_corpus(pairs_path=PAIRS) if generator.corpus else None
        built = []
        for i in range(SAMPLE):
            kwargs = dict(generator.config(), num_docs=n_docs)
            if generator.indexed:
                kwargs["index"] = i
            if corpus is not None:
                kwargs["corpus"] = corpus
            example = generator.build_example(random.Random(f"cmp:{task}:{i}"), **kwargs)
            if example is not None:
                built.append(example)
        print(f"{task} ported   n={n_docs}: {audit.overlap_pair_is_gold(built, spec)}\n")


if __name__ == "__main__":
    main()
