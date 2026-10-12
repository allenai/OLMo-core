"""
Run ``ctc-data`` with its rung sizes taken from a measured calibration instead of ``ctc``'s table.

``ctc`` resolves every rung's document count (oolong: token budget) through
``ctc.data.ladders.docs_for_rung``, with no command-line override. This replaces that one function
for the (task, rung) cells a calibration names -- built by ``measure_eval_ladder.py`` from the eval
data itself -- and then runs the unmodified ``ctc.data.cli``. Every other cell, and every other part
of generation, is ``ctc`` as published.

    CTC_DOCS_OVERRIDE='{"strmatch:2k": 38}' python ctc_calibrated.py build --task strmatch ...
"""

import json
import os
import sys


def main() -> int:
    table = json.loads(os.environ.get("CTC_DOCS_OVERRIDE") or "{}")

    from ctc.data import ladders

    original = ladders.docs_for_rung

    def docs_for_rung(task, rung, *a, **kw):
        key = f"{task}:{str(rung).strip().lower()}"
        if key in table:
            return int(table[key])
        return original(task, rung, *a, **kw)

    ladders.docs_for_rung = docs_for_rung
    if table:
        print(f"[ctc_calibrated] rung sizes from calibration: {table}", flush=True)

    from ctc.data import cli

    return cli.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
