"""Does outlier's topic recovery scale to the ultra-long rungs?

The prefix+tail construction for ``outlier`` needs per-document topic labels: a tail filler document
whose topic ends up with fewer members than the real outlier trio would make the example
unanswerable. The released eval files strip those labels (every ``title`` is ``None``), so
``build_shared_corpus_evals._recover_topics`` recovers them by TF-IDF + agglomerative clustering,
**gated** on exactly reproducing ``meta.category_distribution``. Rows that do not reproduce it are
skipped rather than guessed at.

That gate is the thing that may not scale. At the 32k rung a row holds ~220 documents in ~28
topics; at the 1M rung it holds 7,209 documents in a correspondingly larger topic set, and the gate
demands the clustering reproduce the *entire* multiset of topic sizes exactly.

This measures the gate's success rate and cost per row at the largest rung available locally, so the
extrapolation to 1M rests on numbers rather than on intuition.

    PYTHONPATH=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src \
        python debug/fast_bundle/measure_topic_recovery.py
"""

from __future__ import annotations

import json
import sys
import time

RUNGS = "/scratch/users/prasann/ctc_suite_staged/eval_rungs/outlier"
N_ROWS = 25


def main() -> int:
    from corpus_reasoning.data.build_shared_corpus_evals import _recover_topics

    for rung in (2048, 8192, 32768):
        path = f"{RUNGS}/rung_{rung}.jsonl"
        rows = []
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
                if len(rows) >= N_ROWS:
                    break

        ndocs = len(rows[0]["documents"])
        ncats = rows[0]["meta"]["num_categories"]
        ok, elapsed = 0, 0.0
        for r in rows:
            t0 = time.time()
            try:
                got = _recover_topics(r)
            except (
                Exception
            ) as e:  # noqa: BLE001 - a failure to recover is a datapoint, not a crash
                print(f"    row raised {type(e).__name__}: {e}")
                got = None
            elapsed += time.time() - t0
            ok += got is not None

        print(
            f"rung {rung:>7}  ndocs={ndocs:>5}  topics={ncats:>4}  "
            f"recovered {ok}/{len(rows)} ({ok / len(rows):.0%})  "
            f"{elapsed / len(rows):.2f}s/row"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
